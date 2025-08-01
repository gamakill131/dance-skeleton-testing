import * as poseDetection from '@tensorflow-models/pose-detection';
const BackendURL = import.meta.env.REACT_APP_BACKEND_URL;

export interface LevelData {
    title: string;
    intervals: number[][];
    pose_data: TimestampedPoses[];
}

export interface TimestampedPoses {
    poses: poseDetection.Pose[],
    timestamp: number,
}

export interface LevelCreationData {
    title: string,
    intervals: number[][],
}

class Endpoints {
    getLevel = (objectId: string): Promise<LevelData> => {
        return fetch(`${BackendURL}/levels/${objectId}`)
            .then(response => response.json())
            .then(data => data as LevelData)
            .catch(error => {
                console.error("Error fetching level:", error);
                throw error;
            });
    }

    getOriginalVideo = (objectId: string): Promise<string> => {
        return fetch(`${BackendURL}/levels/getVideo/${objectId}`)
            .then(async response => {
                if (!response.ok) {
                    throw new Error("Network response was not ok");
                }

                const blob = await response.blob();
                const url = URL.createObjectURL(blob);
                return url;
            })
            .catch(error => {
                console.error("Error fetching original video:", error);
                throw error;
            });
    }

    getAnnotatedVideo = (objectId: string): Promise<string> => {
        return fetch(`${BackendURL}/levels/getAnnotatedVideo/${objectId}`)
            .then(response => response.blob())
            .then(blob => URL.createObjectURL(blob))
            .catch(error => {
                console.error("Error fetching annotated video:", error);
                throw error;
            });
    }


    createLevel = (video: File, data: LevelCreationData): Promise<string> => {
        const formData = new FormData();
        formData.append("video", video);
        formData.append("data", JSON.stringify(data));

        return fetch(`${BackendURL}/levels/create`, {
            method: "POST",
            body: formData,
        })
        .then(async response => {
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Network response was not ok: ${errorText}`);
            }
            
            const text = await response.text();
            return text;
        })
        .catch(error => {
            console.error("Error creating level:", error);
            throw error;
        });
    }
}

const instance = new Endpoints();
export default instance;