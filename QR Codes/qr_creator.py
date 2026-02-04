import qrcode
from qrcode.image.styledpil import StyledPilImage
from qrcode.image.styles.moduledrawers import CircleModuleDrawer
from qrcode.image.styles.moduledrawers.pil import RoundedModuleDrawer
from qrcode.image.styles.colormasks import SolidFillColorMask

qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_H)

qr.add_data("https://www.bestcustomapps.dk/Om/")

qr.make(fit=True)

img = qr.make_image(
    image_factory=StyledPilImage,
    module_drawer=CircleModuleDrawer(),      # “points”
    eye_drawer=RoundedModuleDrawer(),        # finder patterns (“eyes”)
    color_mask=SolidFillColorMask(
        back_color=(255, 255, 255),
        front_color=(0, 0, 0),
    ),
    embedded_image_path="logo.png",
)

img.save("qr.png")
