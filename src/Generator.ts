import * as tf from '@tensorflow/tfjs'
import Model from './Model'
import GUI from './GUI'

export default class Gen extends Model {
  constructor(gui: GUI) {
    super(gui)
    this.IMAGE_WIDTH = 32
    this.IMAGE_HEIGHT = 32
    this.IMAGE_CHANNELS = 3
    this.INPUT_SHAPE = [64]
    this.OUTPUT_TYPE = 'image'
    this.build()
  }

  protected build() {
    this.net = tf.sequential()
    if ('add' in this.net) {
      this.net.add(
        tf.layers.dense({
          inputShape: this.INPUT_SHAPE,
          units: 64,
          activation: 'relu',
        })
      )

      this.net.add(
        tf.layers.reshape({
          targetShape: [8, 8, 1],
        })
      )

      this.net.add(
        tf.layers.conv2dTranspose({
          filters: 4,
          kernelSize: 4,
          strides: 2,
          padding: 'same',
          activation: 'relu',
        })
      )

      this.net.add(
        tf.layers.conv2dTranspose({
          filters: 8,
          kernelSize: 4,
          strides: 2,
          padding: 'same',
          activation: 'relu',
        })
      )

      this.net.add(
        tf.layers.conv2dTranspose({
          filters: 3,
          kernelSize: 3,
          strides: 1,
          padding: 'same',
          activation: 'tanh',
        })
      )
    }
  }

  protected seed(batch: number) {
    const std_dev = 3.5
    return tf.randomNormal([batch, ...this.INPUT_SHAPE], 0, std_dev)
  }

  public async doPrediction() {
    return tf.tidy(() => {
      const input = this.seed(1)
      const scale = tf.scalar(0.5)
      const pred = this.net.predict(input) as tf.Tensor
      const squeeze = pred.squeeze().mul(scale).add(scale)
      return squeeze
    })
  }
}
