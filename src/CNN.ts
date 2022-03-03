import * as tf from '@tensorflow/tfjs'
import Model from './Model'
import { MnistData } from './data.js'

export default class CNN extends Model {
  private IMAGE_WIDTH: number
  private IMAGE_HEIGHT: number
  private IMAGE_CHANNELS: number

  public classNames: string[] = [
    'Zero',
    'One',
    'Two',
    'Three',
    'Four',
    'Five',
    'Six',
    'Seven',
    'Eight',
    'Nine',
  ]

  constructor() {
    super()
    this.IMAGE_WIDTH = 28
    this.IMAGE_HEIGHT = 28
    this.IMAGE_CHANNELS = 1
    this.build()
  }

  protected build() {
    this.model = tf.sequential()
    this.model.add(
      tf.layers.conv2d({
        inputShape: [this.IMAGE_WIDTH, this.IMAGE_HEIGHT, this.IMAGE_CHANNELS],
        kernelSize: 5,
        filters: 8,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling',
      })
    )

    this.model.add(
      tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] })
    )

    this.model.add(
      tf.layers.conv2d({
        kernelSize: 5,
        filters: 16,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling',
      })
    )
    this.model.add(
      tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] })
    )

    this.model.add(tf.layers.flatten())

    const NUM_OUTPUT_CLASSES = 10
    this.model.add(
      tf.layers.dense({
        units: NUM_OUTPUT_CLASSES,
        kernelInitializer: 'varianceScaling',
        activation: 'softmax',
      })
    )

    const optimizer = tf.train.adam()
    this.model.compile({
      optimizer: optimizer,
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy'],
    })
  }

  public doPrediction(data: MnistData, testDataSize = 500) {
    const testData = data.nextTestBatch(testDataSize)
    const testxs = testData.xs.reshape([
      testDataSize,
      this.IMAGE_WIDTH,
      this.IMAGE_HEIGHT,
      1,
    ])
    const labels = testData.labels.argMax(-1)
    const preds = this.model.predict(testxs) as tf.Tensor
    const pred = preds.argMax(-1)

    testxs.dispose()
    return [pred, labels]
  }
}
