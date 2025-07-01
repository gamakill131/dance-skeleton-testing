import * as tf from '@tensorflow/tfjs';
import * as poseDetection from '@tensorflow-models/pose-detection';
import type { LevelData, TimestampedPoses } from '~/api/endpoints';
import type { Keypoint } from '@tensorflow-models/pose-detection';

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
                timestamp: videoFrame.timestamp/1000
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
        if(!keypoint.score) {
            continue;
        }
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

        if(!kp1.score || !kp2.score) {
            continue;
        }

        // If both keypoints are confident enough, draw the bone
        if (kp1.score >= SCORE_THRESHOLD && kp2.score >= SCORE_THRESHOLD) {
            ctx.beginPath();
            ctx.moveTo(kp1.x, kp1.y);
            ctx.lineTo(kp2.x, kp2.y);
            ctx.stroke();
        }
    }
}

   /**
     * Compares two poses and returns the root mean square distance between them after translation/scaling normalization
     * @param pose1 - The first pose to compare
     * @param pose2 - The second pose to compare
     * @returns The root mean square distance between the two poses
     */
function comparePoses2D(pose1: poseDetection.Pose, pose2: poseDetection.Pose) {
    // Apply Procrusted Analysis (but only normalize with translation and scaling, not rotation)

    let centroid1: number[] = [0, 0];
    let centroid2: number[] = [0, 0];

    // Calculate the centroid of each pose
    for (const keypoint of pose1.keypoints) {
        centroid1[0] += keypoint.x;
        centroid1[1] += keypoint.y;
    }
    centroid1[0] /= pose1.keypoints.length;
    centroid1[1] /= pose1.keypoints.length;

    for (const keypoint of pose2.keypoints) {
        centroid2[0] += keypoint.x;
        centroid2[1] += keypoint.y;
    }
    centroid2[0] /= pose2.keypoints.length;
    centroid2[1] /= pose2.keypoints.length;

    // Subtract the centroids from each keypoint to get the relative positions
    const relative1: Keypoint[] = pose1.keypoints.map(kp => ({
        x: kp.x - centroid1[0],
        y: kp.y - centroid1[1],
        name: kp.name,
    }));

    const relative2: Keypoint[] = pose2.keypoints.map(kp => ({
        x: kp.x - centroid2[0],
        y: kp.y - centroid2[1],
        name: kp.name,
    }));
    
    // Next, compute the root mean square distance of each pose
    const rmsd1 = Math.sqrt(relative1.reduce((sum, kp) => sum + kp.x * kp.x + kp.y * kp.y, 0) / relative1.length);
    const rmsd2 = Math.sqrt(relative2.reduce((sum, kp) => sum + kp.x * kp.x + kp.y * kp.y, 0) / relative2.length);

    // Scale the keypoints by these scales
    const final1 = relative1.map(kp => ({
        x: kp.x / rmsd1,
        y: kp.y / rmsd1,
        name: kp.name,
    }));

    const final2 = relative2.map(kp => ({
        x: kp.x / rmsd2,
        y: kp.y / rmsd2,
        name: kp.name,
    }));

    // Get the root square difference between the two poses
    let sum = 0;
    for(const kp1 of final1) {
        for(const kp2 of final2) {
            if(kp1.name == kp2.name) {
                sum += (kp1.x - kp2.x) ** 2 + (kp1.y - kp2.y) ** 2;
            }
        }
    }
    return Math.sqrt(sum);
}

function comparePosesByAngles2D(pose1: poseDetection.Pose, pose2: poseDetection.Pose): { [key: string]: number } {
    const angles_to_consider: string[][] = [
        ["left_shoulder", "left_elbow", "left_wrist"],
        ["right_shoulder", "right_elbow", "right_wrist"],
        ["left_hip", "left_knee", "left_ankle"],
        ["right_hip", "right_knee", "right_ankle"],
        ["left_hip", "left_shoulder", "right_shoulder"],
        ["right_hip", "right_shoulder", "left_shoulder"],
        ["left_knee", "left_hip", "right_hip"],
        ["right_knee", "right_hip", "left_hip"],
    ]

    let total_score = 0;

    // Store the scores for each angle. The name for each angle is the name of the second keypoint of the angle
    let all_angle_scores: { [key: string]: number } = {};

    // Calculate the angle between the three points for each angle in the list
    for(const angle of angles_to_consider) {
        const kp1_one = pose1.keypoints.find(kp => kp.name == angle[0]);
        const kp1_two = pose1.keypoints.find(kp => kp.name == angle[1]);
        const kp1_three = pose1.keypoints.find(kp => kp.name == angle[2]);

        const kp2_one = pose2.keypoints.find(kp => kp.name == angle[0]);
        const kp2_two = pose2.keypoints.find(kp => kp.name == angle[1]);
        const kp2_three = pose2.keypoints.find(kp => kp.name == angle[2]);

        if(!kp1_one || !kp1_two || !kp1_three || !kp2_one || !kp2_two || !kp2_three) {
            continue;
        }

        const angle_one = calculateAngleFromPoints(kp1_one.x, kp1_one.y, kp1_two.x, kp1_two.y, kp1_three.x, kp1_three.y);
        const angle_two = calculateAngleFromPoints(kp2_one.x, kp2_one.y, kp2_two.x, kp2_two.y, kp2_three.x, kp2_three.y);

        const angle_difference = Math.abs(angle_one - angle_two);
        const angle_score = 100 - (angle_difference) / Math.PI;

        all_angle_scores[angle[1]] = angle_score;
        total_score += angle_score;
    }

    all_angle_scores["total"] = total_score;

    return all_angle_scores;
}

function calculateAngleFromPoints(x1: number, y1: number, x2: number, y2: number, x3: number, y3: number) {
    const angle = Math.atan2(y2 - y1, x2 - x1) - Math.atan2(y3 - y1, x3 - x1);
    return angle;
}

// Scoring Class
/**
 * Class that handles all the scoring logic for a dance session.
 * 
 * @param levelData - The level data to score
 * @param window_size - The size of the window to score
 * @param intervals - The intervals to score
 */
export class SessionScorer {
    private levelData: LevelData;
    private lastScoredTimestamp: number;
    private window_size: number;
    private timestamp_scores: [number, number][];
    private current_window_average: number;
    private intervals: [number, number][];

    constructor(levelData: LevelData, window_size: number = 5000, intervals: [number, number][] = []) {
        this.levelData = levelData;
        this.lastScoredTimestamp = 0;
        this.window_size = window_size;
        this.current_window_average = 0;
        this.intervals = intervals;
        this.timestamp_scores = [];
    }

    /**
     * Takes in the next user pose and update the scorer
     * @param poses - The next user pose to score
     * @returns The current window average score
     */
    consumePose(poses: TimestampedPoses) {
        // Find the pose in the level data that is closest to the pose in terms of timestamp
        if(this.levelData.pose_data.length == 0) {
            console.log("No level data");
            return this.current_window_average;
        }

        // Binary search to find the closest pose since pose_data is sorted by timestamp
        let left = 0;
        let right = this.levelData.pose_data.length - 1;
        let closestPose: TimestampedPoses = this.levelData.pose_data[0];
        
        while (left <= right) {
            const mid = Math.floor((left + right) / 2);
            const midPose = this.levelData.pose_data[mid];
            
            if (Math.abs(midPose.timestamp - poses.timestamp) < Math.abs(closestPose.timestamp - poses.timestamp)) {
                closestPose = midPose;
            }
            
            if (midPose.timestamp < poses.timestamp) {
                left = mid + 1;
            } else if (midPose.timestamp > poses.timestamp) {
                right = mid - 1;
            } else {
                // Exact match found
                closestPose = midPose;
                break;
            }
        }

        if(closestPose == null) {
            return this.current_window_average;
        }

        // Don't score poses that happened before a pose that was already scored
        if(closestPose.timestamp <= this.lastScoredTimestamp) {
            return this.current_window_average;
        }

        // Score the pose
        const score = comparePoses2D(poses.poses[0], closestPose.poses[0]);
        this.timestamp_scores.push([closestPose.timestamp, score]);
        this.lastScoredTimestamp = closestPose.timestamp;

        // Update the window average (should be the average of all the scored poses from the last scored timestamp to window_size milliseconds in the past)
        const window_start = closestPose.timestamp - this.window_size;
        const window_end = closestPose.timestamp;
        
        // Filter timestamp_scores to get scores within the window
        // Binary search to find the start index of the window
        left = 0;
        right = this.timestamp_scores.length - 1;
        let startIndex = 0;
        
        while (left <= right) {
            const mid = Math.floor((left + right) / 2);
            if (this.timestamp_scores[mid][0] >= window_start) {
                startIndex = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        
        // Binary search to find the end index of the window
        left = startIndex;
        right = this.timestamp_scores.length - 1;
        let endIndex = startIndex;
        
        while (left <= right) {
            const mid = Math.floor((left + right) / 2);
            if (this.timestamp_scores[mid][0] <= window_end) {
                endIndex = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        
        const window_scores = this.timestamp_scores.slice(startIndex, endIndex + 1);
        
        // Calculate window average from the filtered scores
        const window_average = window_scores.length > 0 
            ? window_scores.reduce((sum, [_, score]) => sum + score, 0) / window_scores.length
            : 0;
        this.current_window_average = window_average;
        return window_average;
    }

    /**
     * Computes the scores for each interval (usually 8 counts or something like that)
     * @returns An array of scores for each interval
     */
    computeIntervalScores() {
        let interval_scores: number[] = [];
        for(const interval of this.intervals) {
            const start = interval[0];
            const end = interval[1];
            const window_scores = this.timestamp_scores.filter(([timestamp, _]) => timestamp >= start && timestamp <= end);
            const window_average = window_scores.length > 0 
                ? window_scores.reduce((sum, [_, score]) => sum + score, 0) / window_scores.length
                : 0;
            interval_scores.push(window_average);
        }
        return interval_scores;
    }
        
}
