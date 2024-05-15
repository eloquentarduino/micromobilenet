# MicroMobileNet

Variations of MobileNetV1 meant to run on resource-constrained embedded hardware (a.k.a. microcontrollers).

Blog post at: https://eloquentarduino.com/posts/micro-mobilenet


## Install

```bash
pip install -U micromobilenet
```

## Run

```python
from micromobilenet import PicoMobileNet
from micromobilenet import NanoMobileNet
from micromobilenet import MicroMobileNet
from micromobilenet import MilliMobileNet
from micromobilenet import MobileNet


if __name__ == '__main__':
    net = PicoMobileNet(num_classes=num_classes)
    net.config.learning_rate = 0.001
    net.config.batch_size = 32
    net.config.verbosity = 1
    net.config.checkpoint_min_accuracy = 0.65
    net.config.loss = "categorical_crossentropy"
    net.config.metrics = ["categorical_accuracy"]
    net.config.checkpoint_path = "checkpoints/pico"
    
    net.build()
    net.compile()
    # train_x is of shape (None, 96, 96, 1)
    # train_y is one-hot encoded
    net.fit(train_x, train_y, val_x, val_y, epochs=30)

    print(net.convert.to_cpp())

"""
/**
 * "Compiled" implementation of modified MobileNet
 */
class PicoMobileNet {
public:
    const uint16_t numInputs = 9216;
    const uint16_t numOutputs = 4;
    float outputs[4];
    float arena[6936];
    uint16_t output;
    float proba;

    /**
     *
     */
    MobileNet() : output(0), proba(0) {
        for (uint16_t i = 0; i < numOutputs; i++)
            outputs[i] = 0;
    }

    /**
     *
     * @param input
     */
    uint16_t predict(float *input) {
    ...
};
"""
```

## Deploy

```c++
// sample image is a float[96 * 96] array
#include "sample_image.h"
#include "MobileNet.h"

MobileNet net;

void setup() {
  Serial.begin(115200);
  Serial.println("MobileNet demo");
  
  // no complicated setup! 
}

void loop() {
  size_t start = micros();
  net.predict(sample_image);

  Serial.print("Predicted output = ");
  Serial.println(net.output);
  Serial.print("It took ");
  Serial.print(micros() - start);
  Serial.println(" us to run MobileNet");
  delay(2000);
}
```