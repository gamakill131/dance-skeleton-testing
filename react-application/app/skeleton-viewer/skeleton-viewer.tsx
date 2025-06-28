import { useEffect, useRef, useState } from "react"
import type { Route } from "react-router";
import * as poseDetection from '@tensorflow-models/pose-detection';
import '@tensorflow/tfjs-backend-webgl';
import { createSkeletonVideoTransformer, drawResults } from "./utils"

export interface TimestampedPoses {
  poses: poseDetection.Pose[],
  timestamp: number,
}

interface SkeletonViewerProps {
  reportPoses: (pose: TimestampedPoses) => void,
  useWebcam: boolean,
  mediaStream: MediaStream | null
}

export default function SkeletonViewer({
  reportPoses,
  useWebcam,
  mediaStream
}: SkeletonViewerProps) {
	  // useRef to get a reference to the video element
  const videoRef = useRef<HTMLVideoElement>(null);
  // useState to store and display any error messages
  const [error, setError] = useState<string | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const width = 800;
  const height = 800;

  // This handles streaming webcam to video element and skeleton processing
  useEffect(() => {
    // 2. Define an async function inside the effect to run our setup logic
    const setupAndRun = async () => {
      console.log("Async function running")
      try {

        let stream: MediaStream | null = null;

        // Get the webcam stream
        if(useWebcam) {
          if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
              setError("Your browser does not support accessing the webcam.");
              console.log("Your browser does not support accessing the webcam.")
              return;
          }
          stream = await navigator.mediaDevices.getUserMedia({ video: true });
        }
        // Use the external video stream that should be provided
        else {
          if(!mediaStream) {
            console.log("No Media Stream Provided")
            return;
          }
          stream = mediaStream;
        }
        console.log("Stream Created");

        // Create the processing pipeline
        const videoTrack = stream.getVideoTracks()[0];
        const trackProcessor = new MediaStreamTrackProcessor({ track: videoTrack });
        const trackGenerator = new MediaStreamTrackGenerator({ kind: "video" });
        const transformer: TransformStream = createSkeletonVideoTransformer(reportPoses);

        trackProcessor.readable.pipeThrough(transformer).pipeTo(trackGenerator.writable);
        
        const processedStream = new MediaStream([trackGenerator]);
        console.log("Processed Stream Created")
        if (videoRef.current) {
            videoRef.current.srcObject = processedStream;
            videoRef.current.play();
            console.log("Processed Stream Set On Video Element")
        }

      } catch (err) {
        console.error("Error during setup or processing: ", err);
        setError("Could not access webcam or initialize model. Please grant permission.");
      }
    };

    // Call the async function
    console.log("useEffect running")
    if(videoRef.current && videoRef.current.srcObject) {
      // Don't allow changing the media stream once it has already been set once
      console.log("Media Stream already present")
      return;
    }
    setupAndRun();

    // 5. Cleanup function for both the detector and the stream
    return () => {
      // The detector state is captured in the closure
      if (videoRef.current && videoRef.current.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach(track => track.stop());
        console.log("Webcam stream stopped.");
      }
    };
  }, [mediaStream]);


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
