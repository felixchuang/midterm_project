#include "DA7212.h"
#include "accelerometer_handler.h"
#include "config.h"
#include "magic_wand_model_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "mbed.h"
#include "uLCD_4DGL.h"

#define bufferLength (32)
#define signalLength (208)


DA7212 audio;
Serial pc(USBTX, USBRX);
uLCD_4DGL uLCD(D1, D0, D2); // serial tx, serial rx, reset pin;
DigitalIn button1(SW2);
DigitalIn button2(SW3);
DigitalOut green_led(LED2);
EventQueue queue(32 * EVENTS_EVENT_SIZE);
Thread t1(osPriorityNormal, 120 * 1024 /*120K stack size*/); 
Thread t2(osPriorityNormal, 120 * 1024 /*120K stack size*/); 

int idC = 0;
int sig[signalLength];
int16_t waveform[kAudioTxBufferSize];
char serialInBuffer[bufferLength];
int serialCount = 0;
int song1[42];
int note1[42];
int song2[38];
int note2[38];
int song3[24];
int note3[24];
int load = 0;
int scroll = 0;
int song = 1;


void playSong(void);
void loadSignal(void);
void split(void);
void playNote(int freq, int len);
int PredictGesture(float* output);
void DNN();
void song_sel();
void mode();

void loadSignal(void)
{
  green_led = 0;
  int i = 0;
  serialCount = 0;
  audio.spk.pause();
  while(i < signalLength)
  {
    if(pc.readable())
    {
      serialInBuffer[serialCount] = pc.getc();
      serialCount++;
      if(serialCount == 3)
      {
        serialInBuffer[serialCount] = '\0';
        sig[i] = (int) atof(serialInBuffer);
        uLCD.locate(0, 1);
        uLCD.printf("%.3d", sig[i]);
        serialCount = 0;
        i++;
      }
    }
  }
  green_led = 1;
  load = 1;
}

void split(void)
{
    int i;
    for(i = 0; i < 42; i++)
        song1[i] = sig[i];
    for(i = 42; i < 84; i++)
        note1[i - 42] = sig[i];
    for(i = 84; i < 122; i++)
        song2[i - 84] = sig[i];
    for(i = 122; i < 160; i++)
        note2[i - 122] = sig[i];
    for(i = 160; i < 184; i++)
        song3[i - 160] = sig[i];
    for(i = 184; i < 208; i++)
        note3[i - 184] = sig[i];

}

void playNote(int freq, int len)
{
    for(int i = 0; i < kAudioTxBufferSize; i++){
        waveform[i] = (int16_t) (sin((double)i * 2. * M_PI/(double) (kAudioSampleFrequency / freq)) * ((1<<16) - 1));
    }
    audio.spk.play(waveform, kAudioTxBufferSize);
    //audio.spk.play(0, kAudioTxBufferSize);
}


// Return the result of the last prediction
int PredictGesture(float* output) {
  // How many times the most recent gesture has been matched in a row
  static int continuous_count = 0;
  // The result of the last prediction
  static int last_predict = -1;

  // Find whichever output has a probability > 0.8 (they sum to 1)
  int this_predict = -1;
  for (int i = 0; i < label_num; i++) {
    if (output[i] > 0.8) this_predict = i;
  }

  // No gesture was detected above the threshold
  if (this_predict == -1) {
    continuous_count = 0;
    last_predict = label_num;
    return label_num;
  }

  if (last_predict == this_predict) {
    continuous_count += 1;
  } else {
    continuous_count = 0;
  }
  last_predict = this_predict;

  // If we haven't yet had enough consecutive matches for this gesture,
  // report a negative result
  if (continuous_count < config.consecutiveInferenceThresholds[this_predict]) {
    return label_num;
  }
  // Otherwise, we've seen a positive result, so clear all our variables
  // and report it
  continuous_count = 0;
  last_predict = -1;

  return this_predict;
}
void DNN() {

  // Create an area of memory to use for input, output, and intermediate arrays.
  // The size of this will depend on the model you're using, and may need to be
  // determined by experimentation.
  constexpr int kTensorArenaSize = 60 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];

  // Whether we should clear the buffer next time we fetch data
  bool should_clear_buffer = false;
  bool got_data = false;

  // The gesture index of the prediction
  int gesture_index;

  // Set up logging.
  static tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  const tflite::Model* model = tflite::GetModel(g_magic_wand_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return -1;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  static tflite::MicroOpResolver<6> micro_op_resolver;
  micro_op_resolver.AddBuiltin(
      tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
      tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_MAX_POOL_2D,
                               tflite::ops::micro::Register_MAX_POOL_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,
                               tflite::ops::micro::Register_CONV_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,
                               tflite::ops::micro::Register_FULLY_CONNECTED());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,
                               tflite::ops::micro::Register_SOFTMAX());
    micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_RESHAPE,
                                tflite::ops::micro::Register_RESHAPE(), 1);
  // Build an interpreter to run the model with
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  tflite::MicroInterpreter* interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors
  interpreter->AllocateTensors();

  // Obtain pointer to the model's input tensor
  TfLiteTensor* model_input = interpreter->input(0);
  if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] != config.seq_length) ||
      (model_input->dims->data[2] != kChannelNumber) ||
      (model_input->type != kTfLiteFloat32)) {
    error_reporter->Report("Bad input tensor parameters in model");
    return -1;
  }

  int input_length = model_input->bytes / sizeof(float);

  TfLiteStatus setup_status = SetupAccelerometer(error_reporter);
  if (setup_status != kTfLiteOk) {
    error_reporter->Report("Set up failed\n");
    return -1;
  }

  error_reporter->Report("Set up successful...\n");

  while (true) {

    // Attempt to read new data from the accelerometer
    got_data = ReadAccelerometer(error_reporter, model_input->data.f,
                                 input_length, should_clear_buffer);

    // If there was no new data,
    // don't try to clear the buffer again and wait until next time
    if (!got_data) {
      should_clear_buffer = false;
      continue;
    }

    // Run inference, and report any error
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      error_reporter->Report("Invoke failed on index: %d\n", begin_index);
      continue;
    }

    // Analyze the results to obtain a prediction
    gesture_index = PredictGesture(interpreter->output(0)->data.f);
    printf("%d ", gesture_index);
    if(gesture_index != 1)
        scroll++;
    // Clear the buffer next time we read data
    should_clear_buffer = gesture_index < label_num;
  }
}
void song_sel()
{
    uLCD.cls();
    uLCD.locate(0, 0);
    uLCD.printf("Song select");
    while(button1 == 1){
        switch (scroll % 3)
        {
        case 0:
            uLCD.locate(0, 1);
            uLCD.printf("London Bridge");
            song = 3;
            break;
        case 1:
            uLCD.locate(0, 1);
            uLCD.printf("Twinkle star");
            song = 1;
            break;        
        default:
            uLCD.locate(0, 1);
            uLCD.printf("Butterfly");
            song = 2;
            break;
        } 
        wait(1);       
    }
}
void mode() 
{
    playNote(0, 1);
    int change = 0;
    t2.start(DNN);
    uLCD.cls();
    uLCD.locate(0, 0);
    uLCD.printf("mode selection");
    while(button1 == 1){
        switch (scroll % 3)
        {
        case 0:
            uLCD.locate(0, 1);
            uLCD.printf("Forward ");
            if(song == 1)
                song = 3;
            else
                song--;
            change = 0;
            break;
        case 1:
            uLCD.locate(0, 1);
            uLCD.printf("Backward");
            if(song == 3)
                song = 1;
            else
                song++;
            change = 0;
            break;     
        default:
            uLCD.locate(0, 1);
            uLCD.printf("Change  ");
            change = 1;
            break;
        }
        wait(1);
    }
    if(change){
        song_sel();
        change = 0;
    }
    playSong();
    
}

void playSong()
{
  int i;
  int flag = 1;
  float y, z;
  int score = 0;
        while(1){
            uLCD.cls();
            uLCD.locate(0, 1);
            score = 0;
            switch (song)
            {
            case 1:
                uLCD.printf("Twinkle star ");
                for(i = 0; i < 42 && flag; i++){
                    playNote(song1[i], note1[i]);
                    if(button2 == 0) flag = 0;
                    uLCD.locate(0, 2);
                    if(note1[i] == 1){
                      uLCD.printf("left");
                      if(y - d2 > 0.1){
                        score++;
                      }
                    }
                    else{
                      uLCD.printf("down");
                      if(z - d3 > 0.1){
                        score++;
                      }
                    }
                    uLCD.locate(0, 3);
                    uLCD.printf("Score : %3d", score);
                    wait(0.25);
                }
                song++;
                break;
            case 2:
                uLCD.printf("Butterfly    ");
                for(i = 0; i < 38 && flag; i++){
                    playNote(song2[i], note2[i]);
                    if(button2 == 0) flag = 0;
                    uLCD.locate(0, 2);
                    if(note2[i] == 1){
                      uLCD.printf("left");
                      if(y - d2 > 0.1){
                        score++;
                      }
                    }
                    else{
                      uLCD.printf("down");
                      if(z - d3 > 0.1){
                        score++;
                      }
                    }
                    uLCD.locate(0, 3);
                    uLCD.printf("Score : %3d", score);
                    wait(0.25);
                }
                song++;
                break;
            case 3:
                uLCD.printf("London Bridge");
                for(i = 0; i < 24 && flag; i++){
                    playNote(song3[i], note3[i]);
                    if(button2 == 0) flag = 0;
                    uLCD.locate(0, 2);
                    if(note3[i] == 1){
                      uLCD.printf("left");
                      if(y - d2 > 0.1){
                        score++;
                      }
                      y = d2;
                    }
                    else{
                      uLCD.printf("down");
                      if(z - d3 > 0.1){
                        score++;
                      }
                      z = d3;
                    }
                    uLCD.locate(0, 3);
                    uLCD.printf("Score : %3d", score);
                    wait(0.25);
                } 
                song = 1; 
                break;                  
            default:
                break;
            }
            if(flag == 0){
              mode();
              flag = 1;
            } 
        }
}
int main(int argc, char* argv[]){

    green_led = 1;
    loadSignal();
    if(load){
        split();
        playSong();
    }
    return 0;
}
    