# ------------------------------
# Raymond Ng
# NUS ISS Internship project 2020
# ------------------------------

import argparse
import os
import cv2

def BGRemove(opt):
    # option check
    assert opt.blur_kernel_size % 2 != 0 or opt.blur_kernel_size == 0, 'Kernel size must be an odd number!'

    __bg_algorithm_factory = {
        'MOG2': cv2.createBackgroundSubtractorMOG2(),
        'KNN': cv2.createBackgroundSubtractorKNN(),
        'CNT': cv2.bgsegm.createBackgroundSubtractorCNT(),
        'GMG': cv2.bgsegm.createBackgroundSubtractorGMG(),
        'GSOC': cv2.bgsegm.createBackgroundSubtractorGSOC(),
        'LSBP': cv2.bgsegm.createBackgroundSubtractorLSBP(),
    }

    __thresh_factory = {
        'binary': cv2.THRESH_BINARY,
        'binary_inv': cv2.THRESH_BINARY_INV,
        'trunc': cv2.THRESH_TRUNC,
        '2zero': cv2.THRESH_TOZERO,
        '2zero_inv': cv2.THRESH_TOZERO_INV,
    }

    if opt.save_videos:
        # Create output folder if not exist
        result_root = opt.output_path if opt.output_path !='' else '.'
        os.makedirs(result_root, exist_ok=True)
        # Generate output filename based on input name
        output_mask_file = os.path.join(result_root, '_mask_', os.path.basename(opt.input_video))
        output_det_file = os.path.join(result_root, '_det_', os.path.basename(opt.input_video))
        writer_mask = None
        writer_det = None

    # Get VideoCapture object for input video
    cap = cv2.VideoCapture(opt.input_video)

    # Video details
    n_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create algorithm
    bg_algo = __bg_algorithm_factory[opt.bg_sub_type]

    for i in range(n_frame):
        # Read frame
        grabbed, frame = cap.read()

        # Check if grab frame successful
        if not grabbed:
            print('Read video frame failed!!!')
            break
        
        # Check if gray scale thresholding is required, -1 denotes not required
        if opt.gray_scale_thresh > -1:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, frame = cv2.threshold(frame, opt.gray_scale_thresh, 255, __thresh_factory[opt.thresh_type])
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Apply slight gaussian blur to remove tiny camera shake, does not work for large movement
        if opt.blur_kernel_size > 0:
            frame = cv2.GaussianBlur(frame, (opt.blur_kernel_size, opt.blur_kernel_size), 0)

        # Apply background subtraction
        mask = bg_algo.apply(frame)
        # Object detection using findContours
        _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Draw bounding box
        for c in contours:
            if cv2.contourArea(c) > opt.min_boundingbox_area:
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y,), (x + w, y + h), (0, 0, 255), 5)
        
        # Save videos
        if opt.save_videos:
            if writer_det is None:
                fourcc_mask = cv2.VideoWriter_fourcc(*'X264')
                fourcc_det = cv2.VideoWriter_fourcc(*'X264')
                writer_mask = cv2.VideoWriter(output_mask_file, fourcc_mask, fps, (w, h), True)
                writer_det = cv2.VideoWriter(output_det_file, fourcc_det, fps, (w, h), True)
            writer_mask.write(mask)
            writer_det.write(frame)
        
        # Show video
        cv2.imshow('mask', mask)
        cv2.imshow('detection', frame)

        # Press 'q' to stop
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    if opt.save_videos:
        writer_mask.release()
        writer_det.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Background_Removal.py')
    parser.add_argument('--input-video', type=str, default='test_video', help='Path to input video')
    parser.add_argument('--output-path', type=str, default='output', help='Path for output video')
    parser.add_argument('--blur-kernel-size', type=int, default=0, help='Kernal size for Gaussian blur operation')
    parser.add_argument('--min-boundingbox-area', type=int, default=1000, help='Size of bounding box for contour detection')
    parser.add_argument('--gray-scale-thresh', type=int, default=-1, help='Gray scale threshold value')
    parser.add_argument('--thresh-type', type=str, default='binary', help='Thresholding types')
    parser.add_argument('--save-videos', action='store_true', help='Save output videos')
    parser.add_argument('--bg-sub-type', type=str, default='MOG2', help='Type of background subtract algorithm \
                                                                        [KNN: K-nearest neighbours based] \
                                                                        [MOG2: Gaussian Mixture-based] \
                                                                        [CNT: Counting based] \
                                                                        [GMG: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.228.1735&amp;rep=rep1&amp;type=pdf] \
                                                                        [GSOC: https://docs.opencv.org/3.4/d4/dd5/classcv_1_1bgsegm_1_1BackgroundSubtractorGSOC.html] \
                                                                        [LSBP: Local SVD Binary Pattern]')
    opt = parser.parse_args()
    BGRemove(opt)