/* eslint-disable @typescript-eslint/no-unused-vars */

import * as tf from '@tensorflow/tfjs'
import CNN from './CNN'
import Gen from './Generator'
import GUI from './GUI'
import Editor from './Editor'
import { MnistData } from './data.js'
import StateHandler from './StateHandler'

import { Button } from './types'

import './styles.scss'

async function showExamples(data: MnistData) {
  const examples = data.nextTestBatch(20)
  const numExamples = examples.xs.shape[0]

  for (let i = 0; i < numExamples; i++) {
    const imageTensor = tf.tidy(() => {
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

  const buttons: Array<Button> = [
    {
      selector: '.play-btn',
      eventListener: 'mouseup',
      callback: trainModel,
    },
    {
      selector: '.update-btn',
      eventListener: 'mouseup',
      callback: updateGUI,
    },
    {
      selector: '.predict-btn',
      eventListener: 'mouseup',
      callback: predict,
    },
  ]

  gui.initButtons(buttons)

  //await showExamples(data);

  //const model = await new CNN(gui)
  //await model.warm()

  const model = await new Gen(gui)
  await model.warm()

  const editor = new Editor()
  await gui.initModel(model, editor)
  gui.initOutput(model)
  gui.initChart('loss')

  async function updateGUI() {
    gui.update(model)
  }

  async function trainModel() {
    const playBtn = <HTMLElement>document.querySelector('.play-btn')
    if (model.isTraining) {
      model.isTraining = false
      playBtn.innerText = 'Play'
    } else {
      model.isTraining = true
      playBtn.innerText = 'Pause'
      //await model.train(data)
    }
  }

  async function predict() {
    const pred = await model.doPrediction()
    tf.browser.toPixels(pred as tf.Tensor3D, gui.output)
  }
}

document.addEventListener('DOMContentLoaded', run)
