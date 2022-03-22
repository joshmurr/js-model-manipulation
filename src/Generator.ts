import * as tf from '@tensorflow/tfjs'
import Model from './Model'
import GUI from './GUI'

export default class Gen extends Model {
  private LATENT_DIM: number

  constructor(gui: GUI, latentDim?: number) {
    super(gui)
    this.IMAGE_WIDTH = 32
    this.IMAGE_HEIGHT = 32
    this.IMAGE_CHANNELS = 3
    this.LATENT_DIM = latentDim || 64
    this.OUTPUT_TYPE = 'image'
    this.net = this.build()
  }

  protected build() {
    const G = tf.sequential()
    if ('add' in G) {
      G.add(
        tf.layers.dense({
          inputShape: [this.LATENT_DIM],
          units: this.LATENT_DIM,
          activation: 'relu',
        })
      )

      G.add(
        tf.layers.reshape({
          targetShape: [8, 8, 1],
        })
      )

      G.add(
        tf.layers.conv2dTranspose({
          filters: 16,
          kernelSize: 4,
          strides: 2,
          padding: 'same',
          activation: 'relu',
          kernelInitializer: 'glorotNormal',
        })
      )
      G.add(tf.layers.batchNormalization())

      G.add(
        tf.layers.conv2dTranspose({
          filters: 32,
          kernelSize: 4,
          strides: 2,
          padding: 'same',
          activation: 'relu',
          kernelInitializer: 'glorotNormal',
        })
      )
      G.add(tf.layers.batchNormalization())

      G.add(
        tf.layers.conv2dTranspose({
          filters: 3,
          kernelSize: 3,
          strides: 1,
          padding: 'same',
          activation: 'tanh',
          kernelInitializer: 'glorotNormal',
        })
      )
    }

    const latent = tf.input({ shape: [this.LATENT_DIM] })
    const fakeImage = G.apply(latent)
    return tf.model({
      inputs: [latent],
      outputs: fakeImage as tf.SymbolicTensor,
    })
  }

  protected seed(batch: number) {
    const std_dev = 3.5
    return tf.randomNormal([batch, this.LATENT_DIM], 0, std_dev)
  }

  public async train() {
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

    const BATCH_SIZE = 128

    const trainX = tf.tidy(() => tf.randomNormal([BATCH_SIZE, this.LATENT_DIM]))
    //const targetImage = target || this.generateTargetImage()
    const trainY = this.generateTargetBatch(BATCH_SIZE, false)

    tf.browser
      .toPixels(tf.slice(trainY, [1], 1).squeeze() as tf.Tensor3D)
      .then((imgArr) => {
        const canvas = document.createElement('canvas')
        canvas.width = this.IMAGE_WIDTH
        canvas.height = this.IMAGE_HEIGHT
        const ctx = canvas.getContext('2d')

        const data = new ImageData(imgArr, this.IMAGE_WIDTH, this.IMAGE_HEIGHT)
        ctx.putImageData(data, 0, 0)

        this.gui.displayImage(canvas, 'target')
      })

    //const trainBatch = new Array(BATCH_SIZE).fill(
    //tf.browser.fromPixels(targetImage)
    //)
    //const trainY = tf.stack(trainBatch)

    const learningRate = 0.0025
    const optimizer = tf.train.adam(learningRate)
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

  private randomColour(): string {
    return Math.floor(Math.random() * 16777215).toString(16)
  }

  public generateTargetImage(rand = false): HTMLCanvasElement {
    const canvas = document.createElement('canvas')
    canvas.width = this.IMAGE_WIDTH
    canvas.height = this.IMAGE_HEIGHT
    const ctx = canvas.getContext('2d')

    ctx.fillStyle = rand ? `#${this.randomColour()}` : 'green'
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    ctx.fillStyle = rand ? `#${this.randomColour()}` : 'blue'
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

  public generateTargetBatch(n: number, rand = false) {
    //let sample: HTMLCanvasElement
    const batch: tf.Tensor[] = []
    for (let i = 0; i < n; i++) {
      const canvas = this.generateTargetImage(rand)
      //if (i < 1) sample = canvas
      batch.push(tf.browser.fromPixels(canvas))
    }
    return tf.stack(batch)
  }
}
