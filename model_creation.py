from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from preprosses import image_embaddings
from tensorflow.keras.layers import Conv2D, Input, UpSampling2D
from tensorflow.keras.models import Model

image_paths_x = Path(
    r"C:\Users\ASUS\Desktop\A_GOOD_PROJECT\image_resulation\genrated_images_x"
)
image_paths_y = Path(
    r"C:\Users\ASUS\Desktop\A_GOOD_PROJECT\image_resulation\genrated_images_y"
)

items_x = list(image_paths_x.iterdir())
items_y = list(image_paths_y.iterdir())


def show_two_images(image1, image2):

    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    _, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(image1)
    axes[0].set_title("Image 1")
    axes[0].axis("off")

    axes[1].imshow(image2)
    axes[1].set_title("Image 2")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


def show_dual(items_x, items_y):
    for i in range(len(items_x)):
        show_two_images(image_embaddings(items_x[i]), image_embaddings(items_y[i]))
        print(image_embaddings(items_x[i]).shape, image_embaddings(items_y[i]).shape)
        if i == 1:
            break


def normalization(images):
    embadding = image_embaddings(images)
    normalization_image = embadding.astype(np.float32) / 255.0
    return normalization_image


def denormalization(normalized_image):
    image = normalized_image * 255.0
    image = image.astype(np.uint8)

    return image


def resize_with_padding(image, target_width=1920, target_height=1080):
    """
    Resize image while maintaining aspect ratio.
    Empty areas are filled with black pixels (0, 0, 0).
    """

    h, w = image.shape[:2]

    # Calculate scaling factor
    scale = min(target_width / w, target_height / h)

    new_width = int(w * scale)
    new_height = int(h * scale)

    # Resize image
    resized_image = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_AREA
    )

    # Create black canvas
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    # Calculate center position
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2

    # Put image on black canvas
    canvas[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = (
        resized_image
    )

    return canvas


def get_batches(items_x, items_y, batch_size=8):

    for start in range(0, len(items_x), batch_size):
        end = start + batch_size

        batch_x = []
        batch_y = []

        for x_path, y_path in zip(items_x[start:end], items_y[start:end]):
            # Read images
            x_image = cv2.imread(str(x_path))
            y_image = cv2.imread(str(y_path))

            # Skip invalid images
            if x_image is None or y_image is None:
                continue

            # Resize and add black padding
            x_image = resize_with_padding(x_image)
            y_image = resize_with_padding(y_image)

            # Normalize
            x_image = x_image.astype(np.float32) / 255.0
            y_image = y_image.astype(np.float32) / 255.0

            batch_x.append(x_image)
            batch_y.append(y_image)

        yield (np.array(batch_x, dtype=np.float32), np.array(batch_y, dtype=np.float32))


from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Concatenate,
    MaxPooling2D,
)


def residual_block(x, filters):

    shortcut = x

    # First convolution
    x = Conv2D(filters, kernel_size=3, padding="same")(x)

    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # Second convolution
    x = Conv2D(filters, kernel_size=3, padding="same")(x)

    x = BatchNormalization()(x)

    # Match shortcut channels
    if shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, kernel_size=1, padding="same")(shortcut)

    # Residual connection
    x = Add()([x, shortcut])

    x = Activation("relu")(x)

    return x


def encoder_block(x, filters):

    x = residual_block(x, filters)

    pooled = MaxPooling2D(pool_size=(2, 2))(x)

    return x, pooled


def decoder_block(x, skip, filters):

    x = UpSampling2D(size=(2, 2), interpolation="bilinear")(x)

    x = Concatenate()([x, skip])

    x = residual_block(x, filters)

    return x


def create_model():

    inputs = Input(shape=(None, None, 3))

    # Encoder

    skip1, x = encoder_block(inputs, 32)

    skip2, x = encoder_block(x, 64)

    skip3, x = encoder_block(x, 128)

    skip4, x = encoder_block(x, 256)

    # Bottleneck

    x = residual_block(x, 512)

    # Decoder

    x = decoder_block(x, skip4, 256)

    x = decoder_block(x, skip3, 128)

    x = decoder_block(x, skip2, 64)

    x = decoder_block(x, skip1, 32)

    # Output

    outputs = Conv2D(3, kernel_size=3, padding="same", activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=outputs)

    # Compile

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.Huber(),
        metrics=["mae", "mse"],
    )

    model.summary()

    return model


if __name__ == "__main__":
    model = create_model()

if __name__ == "__main__":
    # normalized_images = []
    # for i in items_x:
    #     normalized_image = normalization(items_x[i])
    #     normalized_images.append(normalized_image)
    # normalized_images = np.array(normalized_images, dtype=np.float32)

    create_model()
    for batch_x, batch_y in get_batches(items_x, items_y, batch_size=4):
        print("X Batch:", batch_x.shape)
        print("Y Batch:", batch_y.shape)

        break

    # show(normalized_image)
