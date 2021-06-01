import datetime
import sys
import csv
import face_recognition
import cv2
import numpy as np
import os
import glob


def train():
    faces_encodings = []
    faces_names = []
    cur_direc = os.getcwd()
    path = os.path.join(cur_direc, 'Photos/')
    list_of_files = [f for f in glob.glob(path + '*.jpeg')]
    number_files = len(list_of_files)
    names = list_of_files.copy()
    # print(names)
    # print(list_of_files)
    for i in range(number_files):
        globals()['image_{}'.format(i)] = face_recognition.load_image_file(list_of_files[i])
        globals()['image_encoding_{}'.format(i)] = face_recognition.face_encodings(globals()['image_{}'.format(i)])[0]
        faces_encodings.append(globals()['image_encoding_{}'.format(i)])
        # Create array of known names
        #     names[i] = names[i].replace(cur_direc, "")
        names[i] = names[i].replace(path, "")
        names[i] = names[i].replace(".jpeg", "")
        names[i] = names[i].replace("1", "")
        faces_names.append(names[i])
    # print(faces_names)
    # print(faces_encodings)
    print("System initiating")
    return faces_names, faces_encodings


def test(faces_names, faces_encodings):
    face_locations = []
    face_encodings = []
    process_this_frame = True
    video_capture = cv2.VideoCapture(0)
    print("Ready!")
    print("Press 'p' for 2 secs to store data.")
    print("Press 'q' once to exit.")
    while True:
        ret, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]
        if process_this_frame:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(faces_encodings, face_encoding)
                name = "Unknown"
                face_distances = face_recognition.face_distance(faces_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = faces_names[best_match_index]
                face_names.append(name)
        process_this_frame = not process_this_frame
        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            # Draw a rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            # Input text label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        # Display the resulting image
        cv2.imshow('Video', frame)
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            sys.exit()
        if cv2.waitKey(1) & 0xFF == ord('p'):
            print(face_names)

            with open('data.csv', 'a', newline='') as csv_file:
                # fieldnames = ['Name', 'Date', 'Time']
                writer_obj = csv.writer(csv_file)
                for i in face_names:
                    timeis = datetime.datetime.now()
                    list = []
                    list.append(i)
                    list.append(str(timeis.strftime("%x")))
                    list.append(str(timeis.strftime("%X")))
                    writer_obj.writerow(list)
            csv_file.close()

            file = open('data.txt', 'a')
            for i in face_names:
                timeis = datetime.datetime.now()
                file.write(i + " entered on " + str(timeis.strftime("%x")) + " at " + str(timeis.strftime("%X")))
            file.write("\n")
            file.close()


def main():
    faces_names, faces_encodings = train()
    with open('data.csv', 'w', newline='') as csvfile:
        fieldnames = ['Name', 'Date', 'Time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    test(faces_names, faces_encodings)


if __name__ == '__main__':
    main()
