/**
 * Class for loading the MoveNet model and various helper methods for 
 * running inference
 * @class
 */
class MoveNet {
    /**
     * Creates a new MoveNet object and calls the load method
     * @constructor
     */
    constructor() {
        this.model = null
        this.load()
    }

    /**
     * Loads the MoveNet model from local storage
     * @async
     */
    async load() {
        this.model = await tf.loadGraphModel('movenet/model.json')
        print('Loaded MoveNet')
    }

    /**
     * Runs inference on the given image and returns the keypoints
     * @param {Canvas} canvas The image to run inference on
     * @returns {Array} Array of keypoints
     */
    getKeypoints(canvas) {
        // We need to convert the video frame to a tensor, and resize it
        // to the input size of the model (192x192)
        const image_tensor = tf.browser.fromPixels(canvas)
        const resized_frame = tf.image.resizeBilinear(image_tensor, [192, 192])
        // Add a dimension for batch size. The model expects a batch of 
        // images, so we need to add a dimension to the tensor. In our 
        // case, the batch size is 1. The shape of the tensor will be 
        // [1, 192, 192, 3]
        const batched_frame = resized_frame.expandDims()
        // Convert from float32 to int32
        const int_frame = batched_frame.toInt()
        // Run inference
        const result = this.model.execute(int_frame)
        // Get keypoints
        const keypoints = result.arraySync()[0][0]
        // Dispose tensors (otherwise memory leaks will happen)
        image_tensor.dispose()
        resized_frame.dispose()
        batched_frame.dispose()
        int_frame.dispose()
        result.dispose()
        // Return keypoints
        return keypoints
    }

    /**
     * Draws the keypoints on the given image
     * @param {Canvas} canvas The image to draw the keypoints on
     * @param {CanvasRenderingContext2D} ctx Canvas's 2D rendering context
     * @param {Array} keypoints Array of keypoints
     */
    drawKeypoints(canvas, ctx, keypoints) {
        for (let i = 0; i < keypoints.length; i+=3){
            const y = keypoints[i] * canvas.width
            const x = keypoints[i+1] * canvas.height
            const score = keypoints[i+2]
            ctx.beginPath()
            ctx.arc(x, y, 5, 0, 2 * Math.PI)
            ctx.fill()
        }
    }
}