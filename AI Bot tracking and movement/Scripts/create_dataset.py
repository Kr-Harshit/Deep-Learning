import os
import cv2
import time
import uuid
from tqdm import tqdm

ROOT_DIR = '.'
paths = {
'DATASET': os.path.join(ROOT_DIR, 'Datasets'),
'GRID': os.path.join(ROOT_DIR, 'Datasets', 'grids'),
'MASK': os.path.join(ROOT_DIR, 'Datasets', 'masks'),
'VIDEOS': os.path.join(ROOT_DIR, 'Datasets', 'videos')
}

def collect_grid_dataset():
    num_imgs = 100
    print('Capturing grid images...')
    cam_url = 0 #Enter Camera Number or Ip webacam URL 
    cap =  cv2.VideoCapture(cam_url)

    for img_num in tqdm(range(num_imgs)):
        if img_num >= num_imgs/2:
            print("Set bot position in 10 secs")
            time.sleep(10)

        ret, frame = cap.read()
        img_name = os.path.join(paths['GRID'], str(uuid.uuid1())+'.jpg')
        cv2.imwrite(img_name, frame)
        time.sleep(2)
    
    print("All images captured")
    cap.release()
    cv2.destroyAllWindows()


def record_video():
    STD_DIMENSIONS =  {
        "480p": (640, 480),
        "720p": (1280, 720),
        "1080p": (1920, 1080),
        "4k": (3840, 2160),
    }

    VIDEO_TYPE = {
        'avi': cv2.VideoWriter_fourcc(*'XVID'),
        'mp4': cv2.VideoWriter_fourcc(*'XVID'),
    }

    def change_res(cap, width, height):
        cap.set(3, width)
        cap.set(4, height)

    def get_dims(cap, res='1080p'):
        width, height = STD_DIMENSIONS["480p"]
        if res in STD_DIMENSIONS:
            width, height = STD_DIMENSIONS[res]
        change_res(cap, width, height)
        return width, height

    #defaults 
    cam_url =  0
    frames_per_second = 24.0
    resolution = '1024p'
    video_type_cv2 = 'avi'
    num_videos = 5

    for i in tqdm(range(num_videos), desc="capturing video..."):
        cap = cv2.VideoCapture(cam_url)
        dims = get_dims(cap, resolution)
        video_name = os.path.join(paths['VIDEOS'], str(uuid.uuid1())+'.'+video_type_cv2)
        out = cv2.VideoWriter(video_name, VIDEO_TYPE[video_type_cv2], frames_per_second, dims)
        start = time.time()
        while True:
            ret, frame = cap.read()
            out.write(frame)
            cv2.imshow(f'Video {i}', frame)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
            if time.time() - start >= 200.0:
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        time.sleep(5)

    print("All video captured")
    

def create_masks_dataset():
    pass

if __name__ == '__main__':
    for path in paths.values():
        if not os.path.exists(path):
            print(path)
            os.makedirs(path)

    # collecting grid images
    collect_grid_dataset()
    # recording videos 
    record_video()