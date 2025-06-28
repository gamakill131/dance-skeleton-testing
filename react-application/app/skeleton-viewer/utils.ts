import * as tf from '@tensorflow/tfjs';
import * as poseDetection from '@tensorflow-models/pose-detection';
import type { TimestampedPoses } from './skeleton-viewer';

// Model and Detector Variables
const SCORE_THRESHOLD = 0.3
const model: poseDetection.SupportedModels = poseDetection.SupportedModels.MoveNet;
const modelType: string = poseDetection.movenet.modelType.SINGLEPOSE_THUNDER;
var detector: poseDetection.PoseDetector | null = null;

// Asynchronously load the detector
tf.ready().then(() => {
    poseDetection.createDetector(model, {modelType}).then(result => {
        detector = result
    });
});

// Stream Transformer that adds skeleton onto the video stream and stores the poses with a function passed in by the caller
export function createSkeletonVideoTransformer(setPoses: (pose: TimestampedPoses) => any): TransformStream {
    console.log("Transforming")
    return new TransformStream({
        async transform(videoFrame: VideoFrame, controller) {
            // If the detector isn't ready for some reason, skip
            if (!detector) {
                controller.enqueue(videoFrame);
                return;
            }
            
            // Lazily initialize canvas and context on the first frame
            const offscreenCanvas = new OffscreenCanvas(videoFrame.displayWidth, videoFrame.displayHeight);
            const ctx = offscreenCanvas.getContext("2d");
      
            if (!ctx) {
                controller.enqueue(videoFrame);
                return;
            }
      
            // Perform detection and drawing
            ctx.drawImage(videoFrame, 0, 0, videoFrame.displayWidth, videoFrame.displayHeight);
            const poses = await detector.estimatePoses(offscreenCanvas as unknown as HTMLCanvasElement, { flipHorizontal: false });
            setPoses({
                poses: poses,
                timestamp: videoFrame.timestamp
            });
            if (poses.length > 0 && model != null) {
                drawResults(poses, ctx, model);
            }
            videoFrame.close();
      
            const newFrame = new VideoFrame(offscreenCanvas, {
                timestamp: videoFrame.timestamp,
                alpha: 'discard'
            });
            controller.enqueue(newFrame);
            console.log("Video Frame Queued");
          }
    });
}

export function drawResults(poses: poseDetection.Pose[], ctx: OffscreenCanvasRenderingContext2D, model: poseDetection.SupportedModels): Boolean {
    for (const pose of poses) {
        drawKeypoints(pose.keypoints, ctx);
        drawSkeleton(pose.keypoints, ctx, model);
    }
    return true;
}

function drawKeypoints(keypoints: poseDetection.Keypoint[], ctx: OffscreenCanvasRenderingContext2D) {
    ctx.fillStyle = 'Red'; // Points color
    ctx.strokeStyle = 'White';
    ctx.lineWidth = 2;

    for (const keypoint of keypoints) {
        if (keypoint.score >= SCORE_THRESHOLD) { // Only draw confident points
            const circle = new Path2D();
            circle.arc(keypoint.x, keypoint.y, 4, 0, 2 * Math.PI);
            ctx.fill(circle);
            ctx.stroke(circle);
        }
    }
}

function drawSkeleton(keypoints: poseDetection.Keypoint[], ctx: OffscreenCanvasRenderingContext2D, model: poseDetection.SupportedModels) {
    ctx.fillStyle = 'White';
    ctx.strokeStyle = '#00ff00'; // Skeleton color
    ctx.lineWidth = 2;

    // Use the official adjacent pairs from the library for the selected model
    const adjacentPairs = poseDetection.util.getAdjacentPairs(model);

    for (const [i, j] of adjacentPairs) {
        const kp1 = keypoints[i];
        const kp2 = keypoints[j];

        // If both keypoints are confident enough, draw the bone
        if (kp1.score >= SCORE_THRESHOLD && kp2.score >= SCORE_THRESHOLD) {
            ctx.beginPath();
            ctx.moveTo(kp1.x, kp1.y);
            ctx.lineTo(kp2.x, kp2.y);
            ctx.stroke();
        }
    }
}
