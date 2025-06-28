import React, { useState, useRef, useEffect } from "react";
import SkeletonViewer, { type TimestampedPoses } from "./skeleton-viewer"; // Adjust path as needed
import * as poseDetection from '@tensorflow-models/pose-detection';

export default function PoseComparisonPage() {
  const [webcamPose, setWebcamPose] = useState<TimestampedPoses[]>([]);
  const [videoPose, setVideoPose] = useState<TimestampedPoses[]>([]);
  const [videoURL, setVideoURL] = useState<string | null>(null);
  const [mediaStream, setMediaStream] = useState<MediaStream | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const secondVideoRef = useRef<HTMLVideoElement>(null);

  // Handle file selection for video
  const handleVideoFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const url = URL.createObjectURL(file);
      setVideoURL(url);

      if(videoRef.current && secondVideoRef.current) {
        videoRef.current.addEventListener("loadeddata", () => {
            if(!videoRef.current || !secondVideoRef.current) return;
            const stream: MediaStream = videoRef.current.captureStream();
            secondVideoRef.current.srcObject = stream;
            setMediaStream(stream);
        });
      }

      // Create a MediaStream for the video file
    //   const videoElement = document.createElement("video");
    //   videoElement.src = url;
    //   videoElement.load();
    //   videoElement.onloadedmetadata = () => {
    //     const stream = videoElement.captureStream();
    //     setMediaStream(stream);
    //   };
    }
  };

  // Function to handle webcam pose reporting
  const handleWebcamPose = (data: TimestampedPoses) => {
    setWebcamPose(prev => [...prev, data]);
  };

  // Function to handle video pose reporting
  const handleVideoPose = (data: TimestampedPoses) => {
    const timestamp = videoRef.current?.currentTime ?? 0;
    setVideoPose(prev => [...prev, data]);
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
      <div style={{ display: "flex", justifyContent: "space-between", gap: "1rem" }}>
        <div style={{ flex: 1 }}>
          <h2>Webcam</h2>
          <SkeletonViewer
            reportPoses={handleWebcamPose}
            useWebcam={true}  // Set the useWebcam prop
            mediaStream={null}
          />
        </div>
        <div style={{ flex: 1 }}>
          <h2>Video</h2>
          <input
            type="file"
            accept="video/*"
            onChange={handleVideoFileChange}
          />
          {(
            // <>
            //   <video
            //     ref={videoRef}
            //     src={videoURL ? videoURL : ""}
            //     controls
            //     style={{ width: "100%", marginTop: "0.5rem" }}
            //   />
            //   <video
            //     ref={secondVideoRef}
            //     controls
            //     style={{ width: "100%", marginTop: "0.5rem" }}
            //   />
            //   <SkeletonViewer
            //     reportPoses={handleVideoPose}
            //     useWebcam={false}  // Set the useWebcam prop as false for video
            //     mediaStream={mediaStream}  // Provide MediaStream for video
            //   />
            // </>
          )}
        </div>
      </div>
    </div>
  );
}