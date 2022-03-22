import * as tf from '@tensorflow/tfjs'
import GUI from './GUI'

import Gen from './Generator'
import Disc from './Discriminator'

export default class DCGAN {
  private learningRate = 0.0002
  private adamBeta1 = 0.5

  constructor(latentDim: number, gui: GUI) {
    const D = new Disc(gui)
    D.compile(this.learningRate, this.adamBeta1)

    const G = new Gen(gui, latentDim)
    const optimizer = tf.train.adam(this.learningRate, this.adamBeta1)
    const combinedModel = this.buildCombinedModel(latentDim, G, D, optimizer)
  }

  buildCombinedModel(
    latentDim: number,
    G: Gen,
    D: Disc,
    optimizer: tf.Optimizer
  ) {
    const latent = tf.input({ shape: [latentDim] })
    let fake = G.net.apply(latent)

    D.net.trainable = false
    fake = D.net.apply(fake)
    const combined = tf.model({
      inputs: latent,
      outputs: fake as tf.SymbolicTensor,
    })
    combined.compile({
      optimizer,
      loss: ['binaryCrossentropy'],
    })
    return combined
  }
}
