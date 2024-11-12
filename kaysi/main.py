from roboflow import Roboflow

rf = Roboflow(api_key="PO84xWBRjBu5TJ10dTyT")
project = rf.workspace().project("odev-mejbg")
model = project.version("1").model
print(model.predict("kaysi/toplu.jpg",confidence=40,overlap=30).save("predictions.jpg"))
