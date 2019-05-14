import cv2
import numpy as np
import imutils
import smtplib
import Contact_details as cd
import imghdr
from email.message import EmailMessage



face_cascade = cv2.CascadeClassifier ( 'haarcascade_frontalface_default.xml' )

capture =  cv2.VideoCapture(0)

img_counter = 0; #licznik do numerowania snapshotów


#wiadomość

msg = EmailMessage()
msg['Subject'] = 'Nieautoryzowany dostep do urzadzenia!'
msg['From'] = cd.EMAIL_SENDER
msg['To'] = cd.EMAIL_RECEIVER
msg.set_content('Kamera wykryla ruch przy Twoim stanowisku, w zalaczniku przesylamy Snapshota z tej sesji.\nZachowaj ostroznosc i zabezpiecz stanowisko.')


while(True):

    ret, frame = capture.read() # sprawdza czy frame jest dobrze odczytana,
                                # ten ret o dziwo jest potrzebny, bez tego nie działa xD
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) # tutaj możemy ewentualnie przekonwertować nasz obraz na grayscale

    faces = face_cascade.detectMultiScale(gray)

    k = cv2.waitKey(1)

    if k%256 == 32:         #jeżeli kliknięty klawisz to spacja
        img_name = "test_{}.jpg".format(img_counter) #nazwanie naszego snapshota
                                                    #format nie ma znaczenia - moze byc dowolny
        cv2.imwrite(img_name, frame) #zapisanie
        print("{} written!".format(img_name)) #potwierdza że Snapshot został wykonany i zapisany

        img_counter += 1  # licznik do nazwy

    if k%256 == 27:
        # dodanie załącznika

        if img_counter > 0 :
            with open("test_{}.jpg".format(img_counter - 1), 'rb') as f:
                file_data = f.read()
                file_type = imghdr.what(f.name)
                file_name = f.name

            msg.add_attachment(file_data, maintype='image', subtype=file_type, filename=file_name)

            # wysyłanie wiaomości
            # with smtplib.SMTP('smtp.gmail.com', 465) as smtp:

            with smtplib.SMTP('smtp.gmail.com', 587) as smtp:
                smtp.ehlo()
                smtp.starttls()
                smtp.ehlo()
                smtp.login(cd.EMAIL_SENDER, cd.PASSWORD)
                smtp.send_message(msg)
                smtp.quit()




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