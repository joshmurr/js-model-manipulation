export default class Editor {
  private container: HTMLElement
  private canvas: HTMLCanvasElement
  private ctx: CanvasRenderingContext2D
  private SHIFT = false
  private SCALE = 30

  constructor() {
    this.buildContainer()
  }

  private buildContainer() {
    this.container = document.createElement('div')
    this.container.id = 'editor'

    this.hideDisplay()

    const close = document.createElement('span')
    close.innerText = 'X'
    close.addEventListener(
      'click',
      function () {
        this.hideDisplay()
      }.bind(this)
    )
    this.container.appendChild(close)

    this.canvas = document.createElement('canvas')
    this.ctx = this.canvas.getContext('2d')
    //this.canvas.width = 5
    //this.canvas.height = 5

    document.body.appendChild(this.container)
  }

  public show(event: MouseEvent) {
    const kernel = <HTMLCanvasElement>event.target
    this.canvas.width = kernel.width
    this.canvas.height = kernel.height
    this.canvas.style.width = `${kernel.width * this.SCALE}px`
    this.canvas.style.height = `${kernel.height * this.SCALE}px`

    this.container.appendChild(this.canvas)

    const ctx = kernel.getContext('2d')
    const imgData = ctx.getImageData(0, 0, kernel.width, kernel.height)

    this.ctx.putImageData(imgData, 0, 0)
    this.showDisplay()
  }

  private draw(event: MouseEvent) {
    const canvas = <HTMLCanvasElement>event.target
    const ctx = canvas.getContext('2d')
    const rect = canvas.getBoundingClientRect()
    const x = Math.floor((event.clientX - rect.left) / this.SCALE)
    const y = Math.floor((event.clientY - rect.top) / this.SCALE)
    const p = ctx.getImageData(x, y, 1, 1)
    const data = p.data
    const adder = this.SHIFT ? 10 : -10
    data[0] += adder
    data[1] += adder
    data[2] += adder
    ctx.putImageData(p, x, y)
  }

  private handleKeyDown(e: KeyboardEvent) {
    if (e.shiftKey) this.SHIFT = true
  }

  private handleKeyUp() {
    this.SHIFT = false
  }

  private showDisplay() {
    this.container.classList.remove('hide')
    this.container.classList.add('show')
    this.canvas.addEventListener('click', (e) => {
      this.draw(e)
    })

    document.addEventListener('keydown', (e) => {
      this.handleKeyDown(e)
    })
    document.addEventListener('keyup', () => {
      this.handleKeyUp()
    })
  }

  private hideDisplay() {
    this.container.classList.add('hide')
    this.container.classList.remove('show')
    if (this.canvas) {
      this.canvas.removeEventListener('click', this.draw)
    }
    document.removeEventListener('keydown', this.handleKeyDown)
    document.removeEventListener('keyup', this.handleKeyUp)
  }
}
