#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <math.h>

#include "TensorFlowLite_ESP32.h"
#include "model_config.h"
#include "model_data.h"
#include "replay_data.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#define REPLAY_MODE 0

Adafruit_MPU6050 imu;
const int SDA_PIN=21, SCL_PIN=22, BUZZER_PIN=4, LED_PIN=2;
const float GRAVITY=9.80665f, GYRO_RAD_TO_DEG=57.2957795f;
const int WINDOW_SAMPLES=100, MODEL_INPUT_CHANNELS=6, STRIDE_SAMPLES=25;
const float MODEL_THRESHOLD=0.60f;
const unsigned long SAMPLE_INTERVAL_MS=20, ALARM_DURATION_MS=1000, ALARM_COOLDOWN_MS=5000;
const float NORMALIZATION_MEAN[6]=NORM_MEAN;
const float NORMALIZATION_STD[6]=NORM_STD;

constexpr int TENSOR_ARENA_SIZE=64*1024;
uint8_t tensor_arena[TENSOR_ARENA_SIZE] __attribute__((aligned(16)));
static tflite::MicroErrorReporter micro_error_reporter;
static tflite::AllOpsResolver resolver;
const tflite::Model* model=nullptr;
tflite::MicroInterpreter* interpreter=nullptr;
TfLiteTensor* input_tensor=nullptr;
TfLiteTensor* output_tensor=nullptr;

float imu_buffer[WINDOW_SAMPLES][MODEL_INPUT_CHANNELS];
int buffer_head=0, total_samples=0, replay_index=0;
bool model_ready=false, alarm_active=false;
unsigned long last_sample_time=0, alarm_start_time=0;

bool i2c_exists(uint8_t address){ Wire.beginTransmission(address); return Wire.endTransmission()==0; }

void scan_i2c(){
  Serial.println("Scanning I2C bus...");
  int n=0;
  for(uint8_t a=1;a<127;a++) if(i2c_exists(a)){
    Serial.print("I2C device found at address: 0x"); if(a<16)Serial.print("0"); Serial.println(a,HEX); n++;
  }
  Serial.print("Total I2C devices found: "); Serial.println(n);
}

void read_live_sample(float s[6]){
  sensors_event_t a,g,t;
  imu.getEvent(&a,&g,&t);
  s[0]=a.acceleration.x/GRAVITY;
  s[1]=a.acceleration.y/GRAVITY;
  s[2]=a.acceleration.z/GRAVITY;
  s[3]=g.gyro.x*GYRO_RAD_TO_DEG;
  s[4]=g.gyro.y*GYRO_RAD_TO_DEG;
  s[5]=g.gyro.z*GYRO_RAD_TO_DEG;
}

void read_replay_sample(float s[6]){
  if(replay_index>=REPLAY_SAMPLES){
    replay_index=0;
    Serial.println("KFall replay sequence restarted");
  }
  for(int c=0;c<6;c++) s[c]=replay_data[replay_index][c];
  replay_index++;
}

int8_t quantize_value(float value){
  int q=(int)round(value/INPUT_SCALE)+INPUT_ZERO_POINT;
  if(q>127)q=127; if(q<-128)q=-128; return (int8_t)q;
}

void fill_input(){
  for(int t=0;t<WINDOW_SAMPLES;t++){
    int bi=(buffer_head+t)%WINDOW_SAMPLES;
    for(int c=0;c<MODEL_INPUT_CHANNELS;c++){
      float z=(imu_buffer[bi][c]-NORMALIZATION_MEAN[c])/NORMALIZATION_STD[c];
      input_tensor->data.int8[t*MODEL_INPUT_CHANNELS+c]=quantize_value(z);
    }
  }
}

float output_probability(int c){
  return (output_tensor->data.int8[c]-output_tensor->params.zero_point)*output_tensor->params.scale;
}

void start_alarm(float alert_p,float fall_p){
  alarm_active=true; alarm_start_time=millis();
  digitalWrite(LED_PIN,HIGH); digitalWrite(BUZZER_PIN,HIGH);
  Serial.println("==============================");
  Serial.println("AI ALARM: POSSIBLE FALL");
  Serial.print("Alert probability: ");Serial.println(alert_p,4);
  Serial.print("Fall probability: ");Serial.println(fall_p,4);
  Serial.println("==============================");
}

void stop_alarm(){
  alarm_active=false; digitalWrite(LED_PIN,LOW); digitalWrite(BUZZER_PIN,LOW); Serial.println("AI ALARM: OFF");
}

void run_inference(){
  fill_input();
  if(interpreter->Invoke()!=kTfLiteOk){ Serial.println("ERROR: TensorFlow Lite model invocation failed"); return; }
  float p0=output_probability(0), p1=output_probability(1), p2=output_probability(2);
  Serial.print("p_nonfall=");Serial.print(p0,4);
  Serial.print(" p_alert=");Serial.print(p1,4);
  Serial.print(" p_fall=");Serial.println(p2,4);
  unsigned long now=millis();
  if((p1>=MODEL_THRESHOLD||p2>=MODEL_THRESHOLD)&&!alarm_active&&now-alarm_start_time>=ALARM_COOLDOWN_MS) start_alarm(p1,p2);
}

bool initialize_model(){
  Serial.println("Loading INT8 TensorFlow Lite model...");
  model=tflite::GetModel(model_data);
  if(model==nullptr){Serial.println("ERROR: model pointer is null");return false;}
  static tflite::MicroInterpreter static_interpreter(model,resolver,tensor_arena,TENSOR_ARENA_SIZE,&micro_error_reporter);
  interpreter=&static_interpreter;
  if(interpreter->AllocateTensors()!=kTfLiteOk){Serial.println("ERROR: AllocateTensors failed");return false;}
  input_tensor=interpreter->input(0); output_tensor=interpreter->output(0);
  if(input_tensor==nullptr||output_tensor==nullptr){Serial.println("ERROR: model tensors unavailable");return false;}
  Serial.print("Input tensor bytes: ");Serial.println(input_tensor->bytes);
  Serial.print("Output tensor bytes: ");Serial.println(output_tensor->bytes);
  Serial.print("Model bytes: ");Serial.println(model_data_len);
  Serial.println("INT8 model loaded successfully");
  return true;
}

void setup(){
  Serial.begin(115200); delay(1500);
  pinMode(BUZZER_PIN,OUTPUT); pinMode(LED_PIN,OUTPUT);
  digitalWrite(BUZZER_PIN,LOW); digitalWrite(LED_PIN,LOW);
  Serial.println("==============================");
  Serial.println("PRE-IMPACT FALL DETECTOR");
  Serial.println("==============================");
  Wire.begin(SDA_PIN,SCL_PIN); Wire.setClock(400000); scan_i2c();
  if(!imu.begin(0x68,&Wire)){Serial.println("ERROR: MPU6050 initialization failed");while(true)delay(1000);}
  Serial.println("MPU6050 READY");
  imu.setAccelerometerRange(MPU6050_RANGE_16_G);
  imu.setGyroRange(MPU6050_RANGE_2000_DEG);
  imu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  model_ready=initialize_model();
  if(!model_ready){while(true){digitalWrite(LED_PIN,!digitalRead(LED_PIN));delay(500);}}
#if REPLAY_MODE
  Serial.println("REPLAY MODE ENABLED");
  Serial.println("Dataset: KFall");
  Serial.println("Trial: S06T20R01");
  Serial.print("Replay samples: ");Serial.println(REPLAY_SAMPLES);
#else
  Serial.println("LIVE MPU6050 MODE ENABLED");
#endif
  Serial.println("SYSTEM READY");
}

void loop(){
  unsigned long now=millis();
  if(now-last_sample_time<SAMPLE_INTERVAL_MS)return;
  last_sample_time=now;
  float sample[6];
#if REPLAY_MODE
  read_replay_sample(sample);
#else
  read_live_sample(sample);
#endif
  for(int c=0;c<MODEL_INPUT_CHANNELS;c++)imu_buffer[buffer_head][c]=sample[c];
  buffer_head=(buffer_head+1)%WINDOW_SAMPLES;
  total_samples++;
  if(total_samples>=WINDOW_SAMPLES&&(total_samples-WINDOW_SAMPLES)%STRIDE_SAMPLES==0)run_inference();
  if(alarm_active&&now-alarm_start_time>=ALARM_DURATION_MS)stop_alarm();
}
