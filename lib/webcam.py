import cv2
import time
import os
import configparser

def webcam(frames_folder, frame_rate = 5, wecam_id = 1):
    cap = cv2.VideoCapture(wecam_id)

    time.sleep(5)

    frame_interval = 1 / frame_rate

    frame_count = 0
    start_time = time.time()

    for i in range(100000000):
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            continue
        
        elapsed_time = time.time() - start_time
        if elapsed_time >= frame_interval:
            frame_count += 1
            start_time = time.time()
            frame_filename = os.path.join(frames_folder, f"{frame_count:08d}.jpg")
            cv2.imwrite(frame_filename, frame)
            #print(f"Saved frame {frame_count}")
        # Exit when 'q' is pressed
        if cv2.waitKey(60) & 0xFF == ord('q'):
            break

    # Release the webcam and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("config.ini", encoding="utf-8")
    output_dir = config["DEFAULT"]["IMGS_FROM_SERVER"]
    webcam("{}/cam0".format(output_dir), frame_rate = 5, wecam_id = 0)