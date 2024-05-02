from ultralytics import YOLO;

model = YOLO("ReciclagemModel.pt")

# Utlizando webcam
# results = model(source=0, show=True, conf=0.3, save=True)

# Utlizando imagem da web
#results = model("https://image.freepik.com/fotos-gratis/papel-amassado_77211-347.jpg", show=True, save=True)

# Utlizando imagem da web
#results = model("https://www.hikaru.restaurant/wp-content/uploads/2019/03/Coca-Cola-lata-33cl.jpg", show=True, save=True)

# Salva e exibi a imagem
# for r in results:
#     im_array = r.plot()  # plot a BGR numpy array of predictions
#     im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
#     im.show()  # show image
#     im.save('results.jpg')

# printa no terminal o nome do objeto identificado
# for result in results:
#     class_id = result.boxes[0].cls[0].item()
#     class_name = result.names[class_id]
#     print("Name:", class_name)