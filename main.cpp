#include "mbed.h"
#include <cmath>
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


DA7212 audio;
int16_t waveform[kAudioTxBufferSize];
EventQueue queue(32 * EVENTS_EVENT_SIZE);
Thread t1(osPriorityNormal, 120 * 1024 /*120K stack size*/); 
Thread t2(osPriorityNormal, 120 * 1024 /*120K stack size*/); 
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


void DNN(int argc, char* argv[]) {


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

  static tflite::MicroOpResolver<5> micro_op_resolver;

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


    // Clear the buffer next time we read data

    should_clear_buffer = gesture_index < label_num;


    // Produce an output

    if (gesture_index < label_num) {

      error_reporter->Report(config.output_message[gesture_index]);

    }

  }

}


int song[42] = {
  261, 261, 392, 392, 440, 440, 392,
  349, 349, 330, 330, 294, 294, 261,
  392, 392, 349, 349, 330, 330, 294,
  392, 392, 349, 349, 330, 330, 294,
  261, 261, 392, 392, 440, 440, 392,
  349, 349, 330, 330, 294, 294, 261};

int noteLength[42] = {
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2};

void playNote(int freq)
{
  for(int i = 0; i < kAudioTxBufferSize; i++)
  {
    waveform[i] = (int16_t) (sin((double)i * 2. * M_PI/(double) (kAudioSampleFrequency / freq)) * ((1<<16) - 1));
  }
  audio.spk.play(waveform, kAudioTxBufferSize);
}

int main(void)
{
  t1.start(DNN);
  t2.start(callback(&queue, &EventQueue::dispatch_forever));

  for(int i = 0; i < 42; i++)
  {
    int length = noteLength[i];
    while(length--)
    {
      // the loop below will play the note for the duration of 1s
      for(int j = 0; j < kAudioSampleFrequency / kAudioTxBufferSize; ++j)
      {
        queue.call(playNote, song[i]);
      }
      if(length < 1) wait(1.0);
    }
  }
}
/*
import numpy as np
import serial
import time

waitTime = 0.1

# generate the waveform table
signalLength = 246
song1 = np.array([
  261, 261, 392, 392, 440, 440, 392,
  349, 349, 330, 330, 294, 294, 261,
  392, 392, 349, 349, 330, 330, 294,
  392, 392, 349, 349, 330, 330, 294,
  261, 261, 392, 392, 440, 440, 392,
  349, 349, 330, 330, 294, 294, 261], dtype=np.int16)
note1 = np.array([
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2], dtype=np.int16)
song2 = np.array([
  392, 330, 300, 349, 294, 294,
  261, 294, 330, 349, 392, 392, 392,
  392, 330, 300, 349, 294, 294,
  261, 330, 392, 392, 330,
  294, 294, 294, 294, 330, 349,
  330, 330 ,330 ,330, 349, 392,
  392, 330, 300, 349, 294, 294,
  261, 330, 392, 392, 261], dtype=np.int16)
note2 = np.array([
  1, 1, 2, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 2, 1, 1, 2,
  1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 2, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 3], dtype=np.int16)
song3 = np.array([
  261, 294, 330, 261, 261, 294, 330, 261,
  330, 349, 392, 330, 349, 392,
  392, 440, 392, 349, 330, 261,
  392, 440, 392, 349, 330, 261,
  294, 196, 261, 294, 196, 261], dtype=np.int16)
note3 = np.array([
  1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 2, 1, 1, 2,
  11, 11, 11, 11, 1, 1,
  11, 11, 11, 11, 1, 1,
  1, 1, 2, 1, 1, 2,], dtype=np.int16)

signalTable = np.append(song1, song2)
signalTable = np.append(signalTable, song3)
signalTable = np.append(signalTable, note1)
signalTable = np.append(signalTable, note2)
signalTable = np.append(signalTable, note3)

# output formatter
formatter = lambda x: "%3d" % x

# send the waveform table to K66F
serdev = '/dev/ttyACM4'
s = serial.Serial(serdev)
print("Sending signal ...")
print("It may take about %d seconds ..." % (int(signalLength * waitTime)))
for data in signalTable:
  s.write(bytes(formatter(data), 'UTF-8'))
  time.sleep(waitTime)
s.close()
print("Signal sended")
*/
/*
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
#include "DA7212.h"

#define bufferLength (32)
#define signalLength (246)

DA7212 audio;
Serial pc(USBTX, USBRX);
uLCD_4DGL uLCD(D1, D0, D2);
DigitalIn button(SW2);
DigitalIn sw3(SW3);
DigitalOut green_led(LED2);
*/
//Thread t1(osPriorityNormal, 120 * 1024 /*120K stack size*/); 
//EventQueue queue(32 * EVENTS_EVENT_SIZE);
//Thread t(osPriorityNormal, 120 * 1024 /*120K stack size*/);;
/*
int flag = 0;
int gesture_index;
int change = 0;
int len[signalLength];
int sig[signalLength];
int16_t waveform[kAudioTxBufferSize];
char serialInBuffer[bufferLength];
int serialCount = 0;
int b1[42] = {
    1, 1, 2, 2, 1, 1, 2,
    1, 1, 2, 2, 1, 1, 2,
    2, 2, 1, 1, 2, 2, 1,
    2, 2, 1, 1, 2, 2, 1,
    1, 1, 2, 2, 1, 1, 2,
    1, 1, 2, 2, 1, 1, 2
};
int b2[47] = {
    1,2,2,1,2,2,
    1,2,1,2,1,1,1,
    1,2,2,1,2,2,
    1,2,1,1,2,
    1,1,1,1,2,2,
    2,2,2,2,1,1,
    1,2,2,1,2,2,
    1,2,1,1,2
};
int b3[34] = {
    1,2,1,2,1,2,1,2,
    2,2,2,1,1,1,
    1,1,2,2,1,1,
    1,1,2,2,1,1,
    2,1,2,1,2,1
};

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

void gesture_thread()
{

    // Create an area of memory to use for input, output, and intermediate arrays.
    // The size of this will depend on the model you're using, and may need to be
    // determined by experimentation.
    constexpr int kTensorArenaSize = 60 * 1024;
    uint8_t tensor_arena[kTensorArenaSize];

    // Whether we should clear the buffer next time we fetch data
    bool should_clear_buffer = false;
    bool got_data = false;

    // Set up logging.
    static tflite::MicroErrorReporter micro_error_reporter;
    tflite::ErrorReporter* error_reporter = &micro_error_reporter;

    // Map the model into a usable data structure. This doesn't involve any
    // copying or parsing, it's a very lightweight operation.
    const tflite::Model* model = tflite::GetModel(g_magic_wand_model_data);

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
  
    int input_length = model_input->bytes / sizeof(float);

    TfLiteStatus setup_status = SetupAccelerometer(error_reporter);

    while(1){

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
        if(gesture_index != 2){
            change++;
            change = change % 3;
        }
        
        // Clear the buffer next time we read data
        should_clear_buffer = gesture_index < label_num;
   }
}

void loadSignal(void)
{
    green_led = 0;
    int i = 0;
    serialCount = 0;
    audio.spk.pause();
    while(i < signalLength){
        if(pc.readable()){
            serialInBuffer[serialCount] = pc.getc();
            printf("%c", serialInBuffer[serialCount]);
            serialCount++;
            if(serialCount == 3){
                serialInBuffer[serialCount] = '\0';
                sig[i] = (int) atof(serialInBuffer);
                serialCount = 0;
                i++;
            }
        }
    }
    green_led = 1;
    flag = 1;
    return;
}

void loadSignalHandler(void) {queue.call(loadSignal);}

void playNote(int freq, int len)
{
    for(int i = 0; i < kAudioTxBufferSize; i++){
        waveform[i] = (int16_t) (sin((double)i * 2. * M_PI/(double) (kAudioSampleFrequency / freq)) * ((1<<16) - 1));
    }
    audio.spk.play(waveform, kAudioTxBufferSize);
    if(len <= 2){
        wait(0.25 * len);
    }
    else{
        wait(0.125);
    }
    audio.spk.play(0, kAudioTxBufferSize);
}


int main(int argc, char* argv[])
{
    int i, song_start, song_end, state = 0, song = 0, count, done;
    int score = 0;
    int b;
    float lasty = 0, lastz = 0;

    green_led = 1;  
    uLCD.cls();
    uLCD.locate(0,1);
    uLCD.printf("loadind songs ...");
    
    
    while(!flag){ 
        uLCD.locate(0,0);
        uLCD.printf("%d",flag);
        if(button == 0){
            loadSignal();
        }
    }
    t1.start(gesture_thread);
    while(1){
        //uLCD.printf("%d", gesture_index);
        if(state == 1){
            done = 0;
            uLCD.locate(0,1);
            uLCD.printf("mode selection : ");
            uLCD.locate(0,3);
            uLCD.printf(" forward songs");
            while(!done){
                
                uLCD.locate(0,0);
                uLCD.printf("%d", change);
                count = change;
                if(count == 0){
                    uLCD.locate(0,3);
                    uLCD.printf(" forward songs");
                }
                else if(count == 1){
                    uLCD.locate(0,3);
                    uLCD.printf("backward songs");
                }
                else if(count ==2){
                    uLCD.locate(0,3);
                    uLCD.printf(" change  songs");
                }
                if(sw3 == 0){
                    done = 1;
                }
            }
            if(count == 0){
                state = 0;
                song--;
                if(song == -1){
                    song = 2;
                }
            }
            else if(count == 1){
                state = 0;
                song++;
                if(song == 3){
                    song = 0;
                }
            }
            else if(count == 2){
                done = 0;
                uLCD.cls();
                uLCD.locate(0,1);
                uLCD.printf("selection songs : ");
                uLCD.locate(0,3);
                uLCD.printf("twinkle star");
                while(!done){
                    
                    uLCD.locate(0,0);
                    uLCD.printf("%d", change);
                    count = change;
                        
                    if(count == 0){
                        uLCD.locate(0,3);
                        uLCD.printf("twinkle star");
                    }
                    else if(count == 1){
                        uLCD.locate(0,3);
                        uLCD.printf("  bees      ");
                    }
                    else if(count ==2){
                        uLCD.locate(0,3);
                        uLCD.printf(" two tigers ");
                    }

                    if(sw3 == 0){
                        done = 1;
                    }
                }
                song = count;
                state = 0;
            }
        }

        if(state == 0){
            score = 0;
            uLCD.cls();
            uLCD.locate(0,1);
            uLCD.printf("Play song : ");
            if(song == 0){
                song_start = 0;
                song_end = 42;
                uLCD.locate(0,3);
                uLCD.printf("twinkle star");
            }
            else if(song == 1){
                song_start = 42;
                song_end = 89;
                uLCD.locate(0,3);
                uLCD.printf("bees");
            }
            else if(song == 2){
                song_start = 89;
                song_end = 123;
                uLCD.locate(0,3);
                uLCD.printf("two tiger");
            }
            for(int i = song_start; i < song_end; ++i){
                lasty = 0;
                lastz = 0;
                playNote(sig[i], sig[i+123]);
                uLCD.locate(0,0);
                uLCD.printf("%d ", sig[i]);
                if(song == 0){
                    b = b1[i];
                }
                else if(song == 1){
                    b = b2[i-42];
                }
                else{
                    b = b3[i-89];
                }
                uLCD.locate(0,5);
                uLCD.printf("beat : %d", b);
                if(b == 1){
                    if(d3 - lastz >= 0.1 || d3 - lastz <= -0.1){
                        score++;
                    }
                    lastz = d3;
                }
                else{
                    if(d2 - lasty >= 0.1 || d2 - lasty <= -0.1){
                        score++;
                    }
                    lasty = d2;
                }
                uLCD.locate(0,7);
                uLCD.printf("score : %d", score);
                wait(0.125);
                if(sw3 == 0){
                    uLCD.cls();
                    state = 1; // mode selection
                    i = song_end;
                }
            }
        }
    }

}
*/