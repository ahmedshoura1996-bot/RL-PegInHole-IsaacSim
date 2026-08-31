import carb

NUCLEUS_ASSET_ROOT_DIR = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"
)


def configure_isaaclab():
    settings = carb.settings.get_settings()

    settings.set(
        "/persistent/isaac/asset_root/cloud",
        NUCLEUS_ASSET_ROOT_DIR,
    )

    return NUCLEUS_ASSET_ROOT_DIR
