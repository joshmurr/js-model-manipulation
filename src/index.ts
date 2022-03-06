/* eslint-disable @typescript-eslint/no-unused-vars */

import * as tf from '@tensorflow/tfjs'
import CNN from './CNN'
import GUI from './GUI'
import { MnistData } from './data.js'
import StateHandler from './StateHandler'

import { Button } from './types'

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

async function run() {
  const data = new MnistData()
  await data.load()

  const gui = new GUI()
  const model = new CNN()

  const buttons: Array<Button> = [
    {
      selector: '.play-btn',
      eventListener: 'mouseup',
      callback: trainModel,
    },
  ]

  gui.initButtons(buttons)

  //await showExamples(data);

  await gui.initModel(model)

  async function trainModel() {
    if (model.isTraining) {
      model.isTraining = false
    } else {
      model.isTraining = true
      await model.train(data, gui)
    }
  }
}

document.addEventListener('DOMContentLoaded', run)
