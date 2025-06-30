import express from 'express';
import multer from 'multer';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import ffmpeg from 'fluent-ffmpeg';
import { CanvasRenderingContext2D, createCanvas, loadImage } from 'canvas';
import * as poseDetection from '@tensorflow-models/pose-detection';
import * as tf from '@tensorflow/tfjs-node';
import { LevelSchema } from '../frontend_models/level_schemas.js';
import validateSchema, { validateID } from '../frontend_models/validate_schema.js';
import Level from '../db_models/level_model.js';
import mongoose from 'mongoose';
import { Readable } from 'stream';
import { isEqualsGreaterThanToken } from 'typescript';

const router = express.Router();
const upload = multer({ dest: 'uploads/' });

const SCORE_THRESHOLD = 0.3;
const MODEL = poseDetection.SupportedModels.MoveNet;
const MODEL_TYPE = poseDetection.movenet.modelType.SINGLEPOSE_THUNDER;

let detector: poseDetection.PoseDetector | null = null;

(async () => {
  await tf.ready();
  detector = await poseDetection.createDetector(MODEL, { modelType: MODEL_TYPE });
})();

interface TimestampedPose {
  timestamp: number;
  poses: poseDetection.Pose[];
}

router.post('/create', upload.single('video'), async (req, res) => {
  if (!req.file) {
    res.status(400).send('No video file uploaded for level');
    return;
  }

  const inputPath = req.file.path;

  try {
    // Parse data JSON
    const dataJSON = JSON.parse(req.body.data);
    if(!validateSchema(dataJSON)) {
      res.status(400).send("Invalid JSON data uploaded for level");
      return;
    }


    // Create a unique temp directory for this request
    const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'poseproc-'));
    const framesDir = path.join(tempDir, 'frames');
    fs.mkdirSync(framesDir);

    const outputVideoPath = path.join(tempDir, 'annotated_output.mp4');
    const outputJSONPath = path.join(tempDir, 'poses.json');

    // Extract frames using ffmpeg
    await new Promise((resolve, reject) => {
      ffmpeg(inputPath)
        .output(`${framesDir}/frame_%06d.png`)
        .on('end', resolve)
        .on('error', reject)
        .run();
    });

    const frameFiles = fs.readdirSync(framesDir).filter(f => f.endsWith('.png')).sort();
    const video_info = await getVideoInfo(inputPath);
    const fps = video_info.fps;
    const poses: TimestampedPose[] = [];

    for (let i = 0; i < frameFiles.length; i++) {
      const framePath = path.join(framesDir, frameFiles[i]);
      const image = await loadImage(framePath);
      const canvas = createCanvas(image.width, image.height);
      const ctx = canvas.getContext('2d');

      ctx.drawImage(image, 0, 0);
      const estPoses = await detector!.estimatePoses(canvas as any, { flipHorizontal: false });
      poses.push({
        timestamp: (i * 1000) / fps,
        poses: estPoses
      });

      drawResults(estPoses, ctx, MODEL);

      fs.writeFileSync(framePath, canvas.toBuffer('image/png'));
    }

    fs.writeFileSync(outputJSONPath, JSON.stringify(poses, null, 2));

    // Combine frames back into video (with the audio)
    await new Promise((resolve, reject) => {
      ffmpeg(`${framesDir}/frame_%06d.png`)
        .inputOptions([`-framerate ${fps}`])
        .input(inputPath) // for audio
        .outputOptions(
          '-map', '0:v',          // video from frames
          '-map', '1:a?',         // audio from original video, if available
          '-c:v', 'libx264',
          '-c:a', 'aac',
          '-pix_fmt', 'yuv420p',
          '-shortest'
        )
        .output(outputVideoPath)
        .on('end', resolve)
        .on('error', reject)
        .run();
    });

    let newLevel = new Level({
      title: dataJSON.title,
      intervals: dataJSON.intervals,
      pose_data: poses,
    });

    newLevel.save().then(doc => {
      if(!mongoose.connection.db) {
        Level.findByIdAndDelete(doc._id);
        res.status(500).send("Error connecting with database to store video files");
        return;
      }
      let bucket = new mongoose.mongo.GridFSBucket(mongoose.connection.db);
      if(!req.file) {
        // This should never run!
        return;
      }

      // Save the original video
      const originalStream = fs.createReadStream(inputPath);
      originalStream.pipe(bucket.openUploadStream("ORIGINAL_"+doc._id.toString()));

      // Save the video annotated with the skeleton
      const skeletonStream = fs.createReadStream(outputVideoPath);
      skeletonStream.pipe(bucket.openUploadStream("ANNOTATED_"+doc._id.toString()));

      // Send the Object ID of the new Level object
      res.send(doc._id.toString());
    });
  } catch (err) {
    console.error('Error during processing:', err);
    res.status(500).send('Failed to process video');
  }
});

router.get('/:id', validateID(), async (req, res) => {
  try {
    const level = await Level.findById(req.params.id);
    if (!level) {
      res.status(404).send('Level not found');
      return;
    }
    res.json(level);
  } catch (err) {
    res.status(500).send('Server error');
  }
});

router.get('/getVideo/:id', async (req, res) => {
  try {
    if(!mongoose.connection.db) {
      res.status(500).send("Error connecting with database");
      return;
    }
    const bucket = new mongoose.mongo.GridFSBucket(mongoose.connection.db);
    const downloadStream = bucket.openDownloadStreamByName("ORIGINAL_" + req.params.id);
    downloadStream.pipe(res);
  } catch (err) {
    res.status(500).send('Error retrieving original video');
  }
});

router.get('/getAnnotatedVideo/:id', async (req, res) => {
  try {
    if(!mongoose.connection.db) {
      res.status(500).send("Error connecting with database");
      return;
    }
    const bucket = new mongoose.mongo.GridFSBucket(mongoose.connection.db);
    const downloadStream = bucket.openDownloadStreamByName("ANNOTATED_" + req.params.id);
    downloadStream.pipe(res);
  } catch (err) {
    res.status(500).send('Error retrieving annotated video');
  }
});

function drawResults(poses: poseDetection.Pose[], ctx: CanvasRenderingContext2D, model: poseDetection.SupportedModels): void {
  for (const pose of poses) {
    drawKeypoints(pose.keypoints, ctx);
    drawSkeleton(pose.keypoints, ctx, model);
  }
}

function drawKeypoints(keypoints: poseDetection.Keypoint[], ctx: CanvasRenderingContext2D) {
  ctx.fillStyle = 'Red';
  ctx.strokeStyle = 'White';
  ctx.lineWidth = 2;

  for (const keypoint of keypoints) {
    if (keypoint.score && keypoint.score >= SCORE_THRESHOLD) {
      ctx.beginPath();
      ctx.arc(keypoint.x, keypoint.y, 4, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();
    }
  }
}

function drawSkeleton(keypoints: poseDetection.Keypoint[], ctx: CanvasRenderingContext2D, model: poseDetection.SupportedModels) {
  ctx.strokeStyle = '#00ff00';
  ctx.lineWidth = 2;

  const adjacentPairs = poseDetection.util.getAdjacentPairs(model);

  for (const [i, j] of adjacentPairs) {
    const kp1 = keypoints[i];
    const kp2 = keypoints[j];

    if(!kp1.score || !kp2.score) {
      continue;
    }

    if (kp1.score >= SCORE_THRESHOLD && kp2.score >= SCORE_THRESHOLD) {
      ctx.beginPath();
      ctx.moveTo(kp1.x, kp1.y);
      ctx.lineTo(kp2.x, kp2.y);
      ctx.stroke();
    }
  }
}

function getVideoInfo(inputPath: string): Promise<{fps: number, width: number | undefined, height:number | undefined}> {
  return new Promise((resolve, reject) => {
    ffmpeg.ffprobe(inputPath, (err, metadata) => {
      if (err) return reject(err);
      const videoStream = metadata.streams.find(s => s.codec_type === 'video');
      if (!videoStream) return reject(new Error('No video stream found'));
      if(!videoStream.avg_frame_rate) return reject(new Error('Average frame rate not given by ffprobe'))

      const [num, den] = videoStream.avg_frame_rate.split('/').map(Number);
      const fps = num / den;
      resolve({ fps, width: videoStream.width, height: videoStream.height });
    });
  });
}

export { router as LevelController };
