import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

base = None

while True:
    cap.grab()
    ret, frame = cap.retrieve()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    if base is None:
        base = gray.copy()
        continue

    # update nền nhanh hơn để đỡ trễ
    base = cv2.addWeighted(base, 0.8, gray, 0.2, 0)

    diff = cv2.absdiff(base, gray)
    thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=1)

    cv2.imshow("Frame", frame)
    cv2.imshow("Thresh", thresh)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
