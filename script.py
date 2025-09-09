import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

os.makedirs("results", exist_ok=True)

img_path = "entrada.jpg"

img_rgb = cv.cvtColor(cv.imread(img_path), cv.COLOR_BGR2RGB)

img_original = img_rgb.copy()

img_1 = img_rgb.copy()

#Define o tamanho do quadrado vermelho
tamanho_quadrado = 100

#Cria o quadrado vermelho no canto superior esquerdo
img_1[0:tamanho_quadrado, 0:tamanho_quadrado] = [255, 0, 0]

height, width, _ = img_rgb.shape

# Define os start points com base no centro da imagem
# Para fazer a linha verde
center_x = width // 2
#Será utilizado o center_y mais para frente, portanto
#Já defini ele aqui em cima
center_y = height // 2
start_point = (center_x, 0)
end_point = (center_x, height)

cv.line(img_1, start_point, end_point, (0, 255, 0), 2)

plt.imshow(img_original)
plt.title("Imagem Original")
plt.show()

plt.imshow(img_1)
plt.title("Imagem Modificada")
plt.show()

img_2 = img_rgb.copy()

azul, verde, vermelho = cv.split(img_2)

#Define o tamanho da figura como 20 polegadas de largura e 
#5 polegadas de altura
fig = plt.figure(figsize=(20, 5))

#Uma linha e três colunas, especificando que é
# primeiro subplot
ax1 = fig.add_subplot(131)
ax1.hist(azul.ravel(), 256, [0, 256], color='blue')
ax1.set_title('Histograma do canal Azul')

# segundo subplot
ax2 = fig.add_subplot(132)
ax2.hist(verde.ravel(), 256, [0, 256], color='green')
ax2.set_title('Histograma do canal Verde')

# terceiro subplot
ax3 = fig.add_subplot(133)
ax3.hist(vermelho.ravel(), 256, [0, 256], color='red')
ax3.set_title('Histograma do canal Vermelho')

plt.show()

# Converte a imagem para tons de cinza
img_2 = cv.cvtColor(img_2, cv.COLOR_BGR2GRAY)

plt.imshow(img_2, cmap='gray')
plt.title("Imagem em Tons de Cinza")
plt.show()

img_3 = img_rgb.copy()

# Desenha o círculo azul no centro da imagem
cv.circle(img_3, (center_x, center_y), 75, (0, 0, 255), -1)
# Começa na width -100 e height - 80
# Vai até o final da imagem
cv.rectangle(img_3, (width-100, height-80), (width, height), (255, 255, 0), -1)
# Faz a linha diagonal vermelha
cv.line(img_3, (0, height), (width, 0), (255, 0, 0), 2)

plt.imshow(img_3)
plt.title("Imagem com círculo azul, retângulo amarelo e linha diagonal vermelha")
plt.show()

img_4 = img_rgb.copy()
img_4_original = img_4.copy()
img_4 = cv.cvtColor(img_4, cv.COLOR_BGR2GRAY)
cv.Canny(img_4, 100, 200, img_4)

# Encontra os contornos na imagem
contours, hierarchy = cv.findContours(img_4, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# Faz retângulos nos contornos
for contour in contours:
    x, y, w, h = cv.boundingRect(contour)
    cv.rectangle(img_4, (x, y), (x + w, y + h), (0, 255, 0), 2)

plt.imshow(img_4)
plt.title("Imagem com Detecção de Bordas")
plt.show()

plt.imshow(img_4_original)
plt.title("Imagem 4 Original")
plt.show()


## Colocar várias imagens lado a lado com subplots, alternando e girando.
img_5 = img_rgb.copy()

img_5_metade = cv.resize(img_5, (width//2, height//2), interpolation=cv.INTER_LINEAR)
img_5_dobro = cv.resize(img_5, (width*2, height*2), interpolation=cv.INTER_LINEAR)

center = (width//2, height//2)
rotation_matrix_45 = cv.getRotationMatrix2D(center, -45, 1.0)
img_5_rot45 = cv.warpAffine(img_5, rotation_matrix_45, (width, height))

rotation_matrix_90 = cv.getRotationMatrix2D(center, -90, 1.0)
img_5_rot90 = cv.warpAffine(img_5, rotation_matrix_90, (width, height))

translation_matrix = np.float32([[1, 0, 50], [0, 1, 100]])
img_5_translacao = cv.warpAffine(img_5, translation_matrix, (width, height))

img_5_flip_horizontal = cv.flip(img_5, 0)
img_5_flip_vertical = cv.flip(img_5, 1)

plt.figure(figsize=(16, 8))

plt.subplot(2, 4, 1)
plt.imshow(img_5)
plt.title('Imagem Original')
plt.axis('off')

plt.subplot(2, 4, 2)
plt.imshow(img_5_metade)
plt.title('Metade do Tamanho')
plt.axis('off')

plt.subplot(2, 4, 3)
plt.imshow(img_5_dobro)
plt.title('Dobro do Tamanho')
plt.axis('off')

plt.subplot(2, 4, 4)
plt.imshow(img_5_rot45)
plt.title('Rotação 45°')
plt.axis('off')

plt.subplot(2, 4, 5)
plt.imshow(img_5_rot90)
plt.title('Rotação 90°')
plt.axis('off')

plt.subplot(2, 4, 6)
plt.imshow(img_5_translacao)
plt.title('Translação (+50x, +100y)')
plt.axis('off')

plt.subplot(2, 4, 7)
plt.imshow(img_5_flip_horizontal)
plt.title('Reflexão Horizontal')
plt.axis('off')

plt.subplot(2, 4, 8)
plt.imshow(img_5_flip_vertical)
plt.title('Reflexão Vertical')
plt.axis('off')

plt.tight_layout()
plt.suptitle('Parte 5 - Transformações Geométricas em Imagens', fontsize=16, y=1.02)
plt.show()