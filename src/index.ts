/* eslint-disable @typescript-eslint/no-unused-vars */

import * as tf from '@tensorflow/tfjs'
import CNN from './CNN'
import GUI from './GUI'
import { MnistData } from './data.js'

import './styles.scss'

async function showExamples(data: MnistData) {
  // Get the examples
  const examples = data.nextTestBatch(20)
  const numExamples = examples.xs.shape[0]

  // Create a canvas element to render each example
  for (let i = 0; i < numExamples; i++) {
    const imageTensor = tf.tidy(() => {
      // Reshape the image to 28x28 px
      return examples.xs
        .slice([i, 0], [1, examples.xs.shape[1]])
        .reshape([28, 28, 1])
    })

    const canvas = document.createElement('canvas')
    canvas.width = 28
    canvas.height = 28
    canvas.style.margin = '4px'
    await tf.browser.toPixels(imageTensor as tf.Tensor2D, canvas)
    document.body.appendChild(canvas)

    imageTensor.dispose()
  }
}

async function train(model: CNN, data: MnistData, gui: GUI) {
  const onEpochEnd = async (epoch: number, logs: tf.Logs) => {
    if (epoch % 10 === 0) {
      console.log('Rendering...')
      await gui.update(model)
    }
  }

  const onTrainEnd = () => {
    console.log('Finito')
  }

  const BATCH_SIZE = 8 // 512
  const TRAIN_DATA_SIZE = 8 // 5500
  const TEST_DATA_SIZE = 8 // 1000

  const [trainXs, trainYs] = tf.tidy(() => {
    const d = data.nextTrainBatch(TRAIN_DATA_SIZE)
    return [d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]), d.labels]
  })

  const [testXs, testYs] = tf.tidy(() => {
    const d = data.nextTestBatch(TEST_DATA_SIZE)
    return [d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]), d.labels]
  })

  return model.model.fit(trainXs, trainYs, {
    batchSize: BATCH_SIZE,
    validationData: [testXs, testYs],
    epochs: 1,
    shuffle: true,
    callbacks: { onEpochEnd, onTrainEnd },
  })
}

async function run() {
  const data = new MnistData()
  await data.load()

  const gui = new GUI()

  //await showExamples(data);

  const model = new CNN()

  await gui.init(model)
  await train(model, data, gui)
}

document.addEventListener('DOMContentLoaded', run)
