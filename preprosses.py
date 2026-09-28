from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

image_paths = Path(
    r"C:\Users\ASUS\.cache\kagglehub\datasets\joe1995\div2k-dataset\versions\1\DIV2K_train_HR\DIV2K_train_HR"
)

image_path = image_paths / "0001.png"

items = list(image_paths.iterdir())

number_of_image = 10


def image_embaddings(image_path):
    return cv2.imread(str(image_path))


def show(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.axis("off")
    plt.show()


def shape_of_image(image):
    return image.shape


def random_value(image):
    h, w = image.shape[:2]

    height = np.random.randint(1, h + 1)
    width = np.random.randint(1, w + 1)

    return width, height


def crop_image(image):
    h, w = image.shape[:2]

    crop_height = np.random.randint(64, h + 1)
    crop_width = np.random.randint(64, w + 1)

    x = np.random.randint(0, w - crop_width + 1)
    y = np.random.randint(0, h - crop_height + 1)

    return image[y : y + crop_height, x : x + crop_width]


def image_genrater(image):
    return [crop_image(image) for _ in range(number_of_image)]


def genrate_more_rotated_image(image):
    final_image_list = []

    for img in image_genrater(image):
        final_image_list.append(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))

        final_image_list.append(cv2.rotate(img, cv2.ROTATE_180))

        final_image_list.append(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE))

    return final_image_list


def saving_the_genrated_images(image_link):

    x_folder = Path(
        r"C:\Users\ASUS\Desktop\A_GOOD_PROJECT\image_resulation\genrated_images_x"
    )

    y_folder = Path(
        r"C:\Users\ASUS\Desktop\A_GOOD_PROJECT\image_resulation\genrated_images_y"
    )

    x_folder.mkdir(parents=True, exist_ok=True)
    y_folder.mkdir(parents=True, exist_ok=True)

    image = image_embaddings(image_link)

    generated_images = genrate_more_rotated_image(image)

    existing_files = list(x_folder.glob("*.jpg"))

    start_number = len(existing_files)

    for j, generated_image in enumerate(generated_images):
        number = start_number + j

        filename = f"{number}.jpg"

        cv2.imwrite(
            str(x_folder / filename),
            generated_image,
        )

        cv2.imwrite(
            str(y_folder / filename),
            generated_image,
        )


def delete_images():
    folders = [
        Path(
            r"C:\Users\ASUS\Desktop\A_GOOD_PROJECT\image_resulation\genrated_images_x"
        ),
        Path(
            r"C:\Users\ASUS\Desktop\A_GOOD_PROJECT\image_resulation\genrated_images_y"
        ),
    ]

    for folder in folders:
        for file in folder.iterdir():
            if file.is_file():
                file.unlink()


def make_xtrain_rough(
    xtrain_path,
    noise_strength=3,
    blur_strength=6,
    pixelation=12,
    tint_rgb=(150, 115, 70),
    tint_strength=0.18,
    scratch_strength=0.25,
    stain_strength=0.20,
    mark_count=15,
):
    xtrain_path = Path(xtrain_path)

    for image_path in xtrain_path.iterdir():
        if not image_path.is_file():
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            continue

        h, w = image.shape[:2]

        scale = max(1, pixelation)

        small_w = max(1, w // scale)
        small_h = max(1, h // scale)

        small = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)

        image = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

        blur = max(1, blur_strength)

        if blur % 2 == 0:
            blur += 1

        image = cv2.GaussianBlur(image, (blur, blur), 0)

        noise = np.random.normal(0, noise_strength, image.shape)

        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        r, g, b = tint_rgb

        tint_color = np.array([b, g, r], dtype=np.uint8)

        tint_image = np.full_like(image, tint_color)

        image = cv2.addWeighted(image, 1 - tint_strength, tint_image, tint_strength, 0)

        overlay = image.copy()

        for _ in range(mark_count):
            x1 = np.random.randint(0, w)
            y1 = np.random.randint(0, h)

            mark_w = np.random.randint(2, max(3, w // 20))

            mark_h = np.random.randint(2, max(3, h // 20))

            x2 = min(w, x1 + mark_w)
            y2 = min(h, y1 + mark_h)

            # Random dark or light mark
            if np.random.random() < 0.6:
                value = np.random.randint(40, 120)

            else:
                value = np.random.randint(180, 240)

            cv2.rectangle(overlay, (x1, y1), (x2, y2), (value, value, value), -1)

        image = cv2.addWeighted(image, 1 - stain_strength, overlay, stain_strength, 0)

        if np.random.random() < scratch_strength:
            scratch_layer = image.copy()

            number_of_scratches = np.random.randint(1, 6)

            for _ in range(number_of_scratches):
                x = np.random.randint(0, w)

                y1 = np.random.randint(0, h)

                y2 = np.random.randint(y1, h)

                scratch_color = np.random.randint(80, 220)

                cv2.line(
                    scratch_layer,
                    (x, y1),
                    (x + np.random.randint(-3, 4), y2),
                    (scratch_color, scratch_color, scratch_color),
                    np.random.randint(1, 3),
                )

            image = cv2.addWeighted(image, 0.85, scratch_layer, 0.15, 0)

        quality = np.random.randint(35, 75)

        _, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])

        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

        cv2.imwrite(str(image_path), image)


if __name__ == "__main__":
    delete_images()

    for image_path in items[:1]:
        saving_the_genrated_images(image_path)

    make_xtrain_rough(
        r"C:\Users\ASUS\Desktop\A_GOOD_PROJECT\image_resulation\genrated_images_x"
    )
