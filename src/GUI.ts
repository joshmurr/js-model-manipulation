import * as tf from '@tensorflow/tfjs'
import Model from './Model'
import Editor from './Editor'
import Chart from './Chart'
import { Button, PixelData, DecodedKernel } from './types'

export default class GUI {
  public container: HTMLElement
  public display: HTMLElement
  public editor: Editor
  public chart: Chart
  private _kernelsToUpdate: Set<string>

  constructor() {
    this._kernelsToUpdate = new Set<string>()
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

  initChart(metrics: string | string[]) {
    this.chart = new Chart(metrics)
  }

  async initModel(model: Model, editor: Editor) {
    this.editor = editor

    const { layerFilters, layerNames } = model.layerInfo

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

          canvas.addEventListener('click', (e) => {
            model.isTraining = false
            editor.show(e, this.setKernel.bind(this))
          })

          row.appendChild(canvas)
        })
        this.display.appendChild(row)
      })
    })

    await this.update(model)
  }

  setKernel(kernelId: string, data: PixelData) {
    this._kernelsToUpdate.add(kernelId)

    const canvas = <HTMLCanvasElement>document.getElementById(kernelId)
    const ctx = canvas.getContext('2d')
    ctx.putImageData(data.p, data.x, data.y)
  }

  async update(model: Model, log?: tf.Logs) {
    await this.drawFilters(model)
    if (log) {
      this.chart.update(log)
      this.chart.draw()
    }

    if (this.editor.needsUpdate) {
      this.updateModel(model)
    }
  }

  updateModel(model: Model) {
    this._kernelsToUpdate.forEach((id) => {
      const canvas = <HTMLCanvasElement>document.getElementById(id)
      const ctx = canvas.getContext('2d')
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
      model.setKernel(this.decodeKernelId(id), imageData)
    })

    this._kernelsToUpdate.clear()
  }

  async drawFilters(model: Model) {
    const { layerFilters, layerNames } = model.layerInfo

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

  private decodeKernelId(kernelId: string): DecodedKernel {
    const splits = kernelId.split('-')
    const layerInfo: DecodedKernel = {
      layer: parseInt(splits[0].slice(1)),
      filter: parseInt(splits[1].slice(1)),
      kernel: parseInt(splits[2].slice(1)),
    }
    return layerInfo
  }
}
