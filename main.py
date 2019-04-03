import cv2
import numpy as np
import imutils


face_cascade = cv2.CascadeClassifier ( 'haarcascade_frontalface_default.xml' )

capture =  cv2.VideoCapture(0)

img_counter = 0; #licznik do numerowania snapshotów

while(True):

    ret, frame = capture.read() # sprawdza czy frame jest dobrze odczytana,
                                # ten ret o dziwo jest potrzebny, bez tego nie działa xD
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) # tutaj możemy ewentualnie przekonwertować nasz obraz na grayscale

    faces = face_cascade.detectMultiScale(gray) 

    k = cv2.waitKey(1)

    if k%256 == 32:         #jeżeli kliknięty klawisz to spacja
        img_name = "test_{}.jpg".format(img_counter) #nazwanie naszego snapshota
        cv2.imwrite(img_name, frame) #zapisanie
        print("{} written!".format(img_name)) #potwierdza że Snapshot został wykonany i zapisany
        img_counter += 1 #licznik do nazwy


    for(x, y, w, h) in faces:   #pętla do wyświetlania figury na twarzy
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        roi_gray = gray[y:y+h, x: x+w]
        roi_color = frame[y:y+h, x: x+w]

    cv2.imshow('Pierwszy przechwyt',frame)   # wyświetlanie obrazu z kamerki (jeżeli byśmy chcieli ten szary obraz,
                                # to byśmy musieli w drugim argumencie wpisać gray zamiast frame


    if k & 0xFF == ord('q'):              # gdy klikniemy q to wyjdzie z pętli
                                            # (dlatego nie klikajcie X w okienku jak chcecie wyłączyć, a q,
                                            # bo jak klikniecie myszką X to wam znów odpali kolejne okienko
                                            # bo będzie dalej w loopie
        break

capture.release()  #zwolnienie kamerki, przestaje przechwytywać obraz
cv2.destroyAllWindows()  #zamyka wszystkie okienka (w naszym przypadku tworzymy je w imshow