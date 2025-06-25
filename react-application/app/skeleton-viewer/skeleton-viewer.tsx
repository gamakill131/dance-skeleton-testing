import { useEffect, useRef, useState } from "react"
import type { Route } from "react-router";
import * as poseDetection from '@tensorflow-models/pose-detection';
import '@tensorflow/tfjs-backend-webgl';
import { drawResults } from "./utils"
import * as tf from '@tensorflow/tfjs';

export default function SkeletonViewer() {
	  // useRef to get a reference to the video element
  const videoRef = useRef<HTMLVideoElement>(null);
  // useState to store and display any error messages
  const [error, setError] = useState<string | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [intervalId, setIntervalId] = useState<NodeJS.Timeout | null>(null);
  const modelRef = useRef<poseDetection.SupportedModels | null>(null);
  const detectorRef = useRef<poseDetection.PoseDetector | null>(null);

  const width = 500;
  const height = 500;

  // This handles streaming webcam to video element and skeleton processing
  useEffect(() => {
    // 2. Define an async function inside the effect to run our setup logic
    const setupAndRun = async () => {
      console.log("Async function running")
      try {
        await tf.ready();
        // 3. Create the detector and store it in state
        modelRef.current = poseDetection.SupportedModels.MoveNet;
        detectorRef.current = await poseDetection.createDetector(modelRef.current);

        console.log("Detector Loaded")

        // --- All the streaming logic now happens only AFTER the detector is ready ---
        
        // Get the webcam stream
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            setError("Your browser does not support accessing the webcam.");
            console.log("Your browser does not support accessing the webcam.")
            return;
        }
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        console.log("Stream Created")

        const transformer = new TransformStream({
          async transform(videoFrame, controller) {
            // If the detector isn't ready for some reason, skip
            if (!detectorRef.current) {
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
            const poses = await detectorRef.current.estimatePoses(offscreenCanvas as unknown as HTMLCanvasElement, { flipHorizontal: false });
            if (poses.length > 0 && modelRef.current != null) {
                drawResults(poses, ctx, modelRef.current);
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

        // Create the processing pipeline
        const videoTrack = stream.getVideoTracks()[0];
        const trackProcessor = new MediaStreamTrackProcessor({ track: videoTrack });
        const trackGenerator = new MediaStreamTrackGenerator({ kind: "video" });

        trackProcessor.readable.pipeThrough(transformer).pipeTo(trackGenerator.writable);
        
        const processedStream = new MediaStream([trackGenerator]);
        console.log("Processed Stream Created")
        if (videoRef.current) {
            videoRef.current.srcObject = processedStream;
            console.log("Processed Stream Set On Video Element")
        }

      } catch (err) {
        console.error("Error during setup or processing: ", err);
        setError("Could not access webcam or initialize model. Please grant permission.");
      }
    };

    // Call the async function
    console.log("useEffect running")
    setupAndRun();

    // 5. Cleanup function for both the detector and the stream
    return () => {
      // The detector state is captured in the closure
      if (detectorRef.current) {
        detectorRef.current.dispose(); // Free up TFJS resources
        console.log("Detector disposed.");
      }
      if (videoRef.current && videoRef.current.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach(track => track.stop());
        console.log("Webcam stream stopped.");
      }
    };
  }, []);


	return (
		<div id="container">
			<video autoPlay={true} id="videoElement" 
      style={{
        backgroundColor: "gray"
      }}
			muted
			playsInline
			ref={videoRef}
      width={width}
      height={height}
      />
      <canvas
        width={width}
        height={height}
        style={{ display: 'block' }}
        ref={canvasRef}
      />
		</div>
	);
}
