import * as tf from '@tensorflow/tfjs'

interface TidiedLogs {
  [key: string]: number[]
}

export default class Chart {
  private container: HTMLElement
  private canvas: HTMLCanvasElement
  private ctx: CanvasRenderingContext2D
  private tidiedLogs: TidiedLogs
  private metrics: string[]

  constructor(metrics: string | string[], parent: HTMLElement) {
    this.initChart(parent)

    if (typeof metrics === 'string') metrics = [metrics]
    this.metrics = metrics
    this.tidiedLogs = {}
    metrics.forEach((metric) => (this.tidiedLogs[metric] = []))
  }

  private initChart(parent: HTMLElement) {
    this.container = document.createElement('div')
    //this.container.id = 'chart'
    this.container.classList.add('chart-container')

    this.canvas = document.createElement('canvas')
    this.canvas.width = 300
    this.canvas.height = 150
    this.canvas.classList.add('chart')
    this.container.appendChild(this.canvas)

    this.ctx = this.canvas.getContext('2d')

    parent.appendChild(this.container)
  }

  public update(log: tf.Logs) {
    this.metrics.forEach((metric) => this.tidiedLogs[metric].push(log[metric]))
  }

  public draw() {
    this.metrics.forEach((metric) => {
      const data = this.tidiedLogs[metric]
      const xScale = this.canvas.width / data.length
      const yMin = Math.min(...data)
      const yMax = Math.max(...data)
      const yRange = yMax - yMin
      const yScale = this.canvas.height / yRange
      this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height)
      this.ctx.beginPath()
      this.ctx.moveTo(0, 0)
      data.forEach((p, i) => {
        const x = i * xScale
        const y = this.canvas.height - p * yScale

        this.ctx.lineTo(x, y)
      })
      this.ctx.stroke()
    })
  }
}
