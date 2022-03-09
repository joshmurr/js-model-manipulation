import * as tf from '@tensorflow/tfjs'
import Model from './Model'
import { MnistData } from './data.js'
import GUI from './GUI'

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
    this.net = tf.sequential()
    if ('add' in this.net) {
      this.net.add(
        tf.layers.conv2d({
          inputShape: [
            this.IMAGE_WIDTH,
            this.IMAGE_HEIGHT,
            this.IMAGE_CHANNELS,
          ],
          kernelSize: 5,
          filters: 8,
          strides: 1,
          activation: 'relu',
          kernelInitializer: 'varianceScaling',
        })
      )

      this.net.add(
        tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] })
      )

      this.net.add(
        tf.layers.conv2d({
          kernelSize: 5,
          filters: 16,
          strides: 1,
          activation: 'relu',
          kernelInitializer: 'varianceScaling',
        })
      )
      this.net.add(
        tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] })
      )

      this.net.add(tf.layers.flatten())

      const NUM_OUTPUT_CLASSES = 10
      this.net.add(
        tf.layers.dense({
          units: NUM_OUTPUT_CLASSES,
          kernelInitializer: 'varianceScaling',
          activation: 'softmax',
        })
      )
    }
  }

  async train(data: MnistData, gui: GUI) {
    const onEpochEnd = async (epoch: number) => {
      if (this.training === false) {
        this.net.stopTraining = true
      }

      if (epoch % 10 === 0) {
        console.log(`Epoch: ${epoch}, Rendering...`)
        await gui.update(this)
      }
    }

    const onTrainEnd = () => {
      console.log('Finished training.')
    }

    const BATCH_SIZE = 8 // 512
    const TRAIN_DATA_SIZE = 8 // 5500
    const TEST_DATA_SIZE = 8 //1000

    const [trainXs, trainYs] = tf.tidy(() => {
      const d = data.nextTrainBatch(TRAIN_DATA_SIZE)
      return [d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]), d.labels]
    })

    const [testXs, testYs] = tf.tidy(() => {
      const d = data.nextTestBatch(TEST_DATA_SIZE)
      return [d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]), d.labels]
    })

    const optimizer = tf.train.adam()
    this.net.compile({
      optimizer: optimizer,
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy'],
    })

    await this.net.fit(trainXs, trainYs, {
      batchSize: BATCH_SIZE,
      validationData: [testXs, testYs],
      epochs: 1e9,
      shuffle: true,
      callbacks: { onEpochEnd, onTrainEnd },
    })

    await this.net.save('localstorage://cnn')

    this.net.dispose()
    this.net = null
    tf.disposeVariables()

    this.net = await tf.loadLayersModel('localstorage://cnn')
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
    const preds = this.net.predict(testxs) as tf.Tensor
    const pred = preds.argMax(-1)

    testxs.dispose()
    return [pred, labels]
  }
}
