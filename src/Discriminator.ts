import * as tf from '@tensorflow/tfjs'
import Model from './Model'
import GUI from './GUI'

export default class Disc extends Model {
  private NUM_OUTPUT_CLASSES: number

  constructor(gui: GUI) {
    super(gui)
    this.IMAGE_WIDTH = 32
    this.IMAGE_HEIGHT = 32
    this.IMAGE_CHANNELS = 3
    this.INPUT_SHAPE = [
      this.IMAGE_WIDTH,
      this.IMAGE_HEIGHT,
      this.IMAGE_CHANNELS,
    ]
    this.OUTPUT_TYPE = 'classification'
    this.NUM_OUTPUT_CLASSES = 2
    this.net = this.build()
  }

  protected build() {
    const D = tf.sequential()
    if ('add' in D) {
      D.add(
        tf.layers.conv2d({
          inputShape: [
            this.IMAGE_WIDTH,
            this.IMAGE_HEIGHT,
            this.IMAGE_CHANNELS,
          ],
          kernelSize: 3,
          filters: 8,
          strides: 2,
          padding: 'same',
        })
      )
      D.add(tf.layers.leakyReLU({ alpha: 0.2 }))
      D.add(tf.layers.dropout({ rate: 0.3 }))

      D.add(
        tf.layers.conv2d({
          kernelSize: 3,
          filters: 16,
          strides: 1,
          padding: 'same',
        })
      )
      D.add(tf.layers.leakyReLU({ alpha: 0.2 }))
      D.add(tf.layers.dropout({ rate: 0.3 }))

      D.add(
        tf.layers.conv2d({
          kernelSize: 3,
          filters: 16,
          strides: 2,
          padding: 'same',
        })
      )
      D.add(tf.layers.leakyReLU({ alpha: 0.2 }))
      D.add(tf.layers.dropout({ rate: 0.3 }))

      D.add(
        tf.layers.conv2d({
          kernelSize: 3,
          filters: 16,
          strides: 1,
          padding: 'same',
        })
      )
      D.add(tf.layers.leakyReLU({ alpha: 0.2 }))
      D.add(tf.layers.dropout({ rate: 0.3 }))

      D.add(tf.layers.flatten())

      const image = tf.input({
        shape: [this.IMAGE_WIDTH, this.IMAGE_HEIGHT, this.IMAGE_CHANNELS],
      })
      const features = D.apply(image)

      const realnessScore = tf.layers
        .dense({ units: 1, activation: 'sigmoid' })
        .apply(features)

      return tf.model({
        inputs: image,
        outputs: realnessScore as tf.SymbolicTensor,
      })
    }
  }

  public compile(learningRate: number, beta: number) {
    this.net.compile({
      optimizer: tf.train.adam(learningRate, beta),
      loss: ['binaryCrossentropy'],
    })
  }
}
