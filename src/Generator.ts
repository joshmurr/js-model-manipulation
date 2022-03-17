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
          filters: 8,
          kernelSize: 4,
          strides: 2,
          padding: 'same',
          activation: 'relu',
        })
      )

      this.net.add(
        tf.layers.conv2dTranspose({
          filters: 16,
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

  public async train(target?: HTMLCanvasElement) {
    const onEpochEnd = async (epoch: number, log: tf.Logs) => {
      if (this.training === false) {
        this.net.stopTraining = true
      }

      if (epoch % 10 === 0) {
        console.log(`Epoch: ${epoch}, Rendering...`)
        await this.gui.update(this, log)
        const pred = await this.doPrediction()
        tf.browser.toPixels(pred as tf.Tensor3D, this.gui.output.modelOutput)
      }
    }

    const onTrainEnd = () => {
      console.log('Finished training.')
    }

    const BATCH_SIZE = 1

    const trainX = tf.tidy(() =>
      tf.randomNormal([BATCH_SIZE, ...this.INPUT_SHAPE])
    )
    const targetImage = target || this.generateTargetImage()

    this.gui.displayImage(targetImage, 'target')

    const trainY = tf.browser.fromPixels(targetImage).expandDims(0)

    const optimizer = tf.train.adam()
    this.net.compile({
      optimizer: optimizer,
      loss: 'meanSquaredError',
    })

    await this.net.fit(trainX, trainY, {
      batchSize: BATCH_SIZE,
      epochs: 1e9,
      callbacks: { onEpochEnd, onTrainEnd },
    })

    await this.net.save('localstorage://generator')

    this.net.dispose()
    this.net = null
    tf.disposeVariables()

    this.net = await tf.loadLayersModel('localstorage://generator')
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

  public generateTargetImage(): HTMLCanvasElement {
    const canvas = document.createElement('canvas')
    canvas.width = this.IMAGE_WIDTH
    canvas.height = this.IMAGE_HEIGHT
    const ctx = canvas.getContext('2d')

    ctx.fillStyle = 'green'
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    ctx.fillStyle = 'blue'
    ctx.beginPath()
    ctx.arc(
      canvas.width / 2,
      canvas.height / 2,
      canvas.width / 2 - 4,
      0,
      Math.PI * 2
    )
    ctx.fill()

    return canvas
  }
}
