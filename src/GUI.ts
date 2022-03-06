import * as tf from '@tensorflow/tfjs'
import Model from './Model'
import { LayerFilters, Button } from './types'

export default class GUI {
  public container: HTMLElement
  public display: HTMLElement

  constructor() {
    this.container = document.getElementById('container')
    this.display = document.createElement('div')
    this.display.id = 'display'
    this.container.appendChild(this.display)
  }

  initButtons(buttons: Array<Button>) {
    buttons.forEach((button) => {
      const { selector, eventListener, callback } = button
      const buttonEl = document.querySelector(selector)
      buttonEl.addEventListener(eventListener, callback)
    })
  }

  async initModel(model: Model) {
    const layers: Array<tf.layers.Layer> = await model.getLayers()
    const layerFilters: LayerFilters = await model.getFilters(layers)
    const layerNames = Object.keys(layerFilters)

    layerNames.forEach((layerName, l_id) => {
      const filterArray = layerFilters[layerName]
      const title = document.createElement('h4')
      title.innerText = layerName
      this.display.appendChild(title)

      filterArray.forEach((filter, f_id) => {
        const row = document.createElement('div')
        row.classList.add('filter-row')
        filter.forEach((kernel, k_id) => {
          const [w, h] = kernel.shape
          const canvas = document.createElement('canvas')
          canvas.width = w
          canvas.height = h
          canvas.id = this.getKernelId(l_id, f_id, k_id)

          row.appendChild(canvas)
        })
        this.display.appendChild(row)
      })
    })
  }

  async update(model: Model) {
    const layers: Array<tf.layers.Layer> = await model.getLayers()
    const layerFilters: LayerFilters = await model.getFilters(layers)
    const layerNames = Object.keys(layerFilters)

    layerNames.forEach((layerName, l_id) => {
      const filterArray = layerFilters[layerName]
      filterArray.forEach((filter, f_id) => {
        filter.forEach((kernel, k_id) => {
          const kernel_id = this.getKernelId(l_id, f_id, k_id)
          const canvas = <HTMLCanvasElement>document.getElementById(kernel_id)
          const ctx = canvas.getContext('2d')
          const [w, h] = kernel.shape
          const img = new ImageData(w, h)
          const data = kernel.dataSync()
          for (let x = 0; x < w; x++) {
            for (let y = 0; y < h; y++) {
              const ix = (y * w + x) * 4
              const iv = y * w + x
              img.data[ix + 0] = Math.floor(127 * (data[iv] + 1))
              img.data[ix + 1] = Math.floor(127 * (data[iv] + 1))
              img.data[ix + 2] = Math.floor(127 * (data[iv] + 1))
              img.data[ix + 3] = 255
            }
          }
          ctx.putImageData(img, 0, 0)
        })
      })
    })
  }

  private getKernelId(l_id: number, f_id: number, k_id: number): string {
    return `l${l_id}-f${f_id}-k${k_id}`
  }
}
