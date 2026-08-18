import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import io

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Image Processing Visualizer",
    page_icon="🖼️",
    layout="wide"
)

st.title("🖼️ Interactive Image Processing Visualizer")

st.write(
    "Upload an image, select an image processing operation, "
    "view its kernel, pixel representation and processed output."
)

# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded_file is not None:

    # --------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------

    image = Image.open(uploaded_file).convert("RGB")

    img = np.array(image)

    # Convert RGB to OpenCV BGR
    img_bgr = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2BGR
    )

    # Convert image to grayscale
    gray = cv2.cvtColor(
        img_bgr,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------
    # INPUT IMAGE
    # --------------------------------------------------

    st.subheader("1. Input Image")

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            image,
            caption="Original Image",
            use_container_width=True
        )

    with col2:

        st.image(
            gray,
            caption="Grayscale Image",
            use_container_width=True
        )

    # --------------------------------------------------
    # INPUT PIXEL REPRESENTATION
    # --------------------------------------------------

    st.subheader("2. Input Pixel Representation")

    st.write(
        f"Image size: **{gray.shape[1]} × {gray.shape[0]} pixels**"
    )

    st.write(
        "The image is represented as a grayscale pixel matrix "
        "where each value ranges from 0 to 255."
    )

    # Display limited pixel matrix
    max_size = 15

    pixel_height = min(
        gray.shape[0],
        max_size
    )

    pixel_width = min(
        gray.shape[1],
        max_size
    )

    pixel_matrix = gray[
        :pixel_height,
        :pixel_width
    ]

    st.write(
        f"Showing first {pixel_height} × "
        f"{pixel_width} pixels:"
    )

    pixel_df = pd.DataFrame(
        pixel_matrix
    )

    st.dataframe(
        pixel_df,
        use_container_width=True
    )

    # --------------------------------------------------
    # OPERATION SELECTION
    # --------------------------------------------------

    st.subheader("3. Select Image Processing Operation")

    operation = st.selectbox(
        "Choose an operation:",
        [
            "Mean Filter",
            "Gaussian Filter",
            "Median Filter",
            "Sobel Operator",
            "Laplacian Operator",
            "Prewitt Operator"
        ]
    )

    # --------------------------------------------------
    # KERNEL SIZE
    # --------------------------------------------------

    if operation in [
        "Mean Filter",
        "Gaussian Filter",
        "Median Filter"
    ]:

        kernel_size = st.selectbox(
            "Select Kernel Size:",
            [3, 5, 7]
        )

    else:

        kernel_size = 3

    # --------------------------------------------------
    # PROCESS IMAGE
    # --------------------------------------------------

    kernel = None

    # --------------------------------------------------
    # MEAN FILTER
    # --------------------------------------------------

    if operation == "Mean Filter":

        kernel = np.ones(
            (
                kernel_size,
                kernel_size
            ),
            dtype=np.float32
        ) / (
            kernel_size * kernel_size
        )

        result = cv2.filter2D(
            gray,
            -1,
            kernel
        )

    # --------------------------------------------------
    # GAUSSIAN FILTER
    # --------------------------------------------------

    elif operation == "Gaussian Filter":

        kernel_1d = cv2.getGaussianKernel(
            kernel_size,
            0
        )

        kernel = kernel_1d @ kernel_1d.T

        result = cv2.GaussianBlur(
            gray,
            (
                kernel_size,
                kernel_size
            ),
            0
        )

    # --------------------------------------------------
    # MEDIAN FILTER
    # --------------------------------------------------

    elif operation == "Median Filter":

        kernel = np.ones(
            (
                kernel_size,
                kernel_size
            ),
            dtype=np.uint8
        )

        result = cv2.medianBlur(
            gray,
            kernel_size
        )

    # --------------------------------------------------
    # SOBEL OPERATOR
    # --------------------------------------------------

    elif operation == "Sobel Operator":

        sobel_x_kernel = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ])

        sobel_y_kernel = np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ])

        sobel_x = cv2.Sobel(
            gray,
            cv2.CV_64F,
            1,
            0,
            ksize=3
        )

        sobel_y = cv2.Sobel(
            gray,
            cv2.CV_64F,
            0,
            1,
            ksize=3
        )

        magnitude = cv2.magnitude(
            np.float32(sobel_x),
            np.float32(sobel_y)
        )

        result = cv2.convertScaleAbs(
            magnitude
        )

        kernel = sobel_x_kernel

    # --------------------------------------------------
    # LAPLACIAN OPERATOR
    # --------------------------------------------------

    elif operation == "Laplacian Operator":

        kernel = np.array([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ])

        laplacian = cv2.filter2D(
            gray,
            cv2.CV_64F,
            kernel
        )

        result = cv2.convertScaleAbs(
            laplacian
        )

    # --------------------------------------------------
    # PREWITT OPERATOR
    # --------------------------------------------------

    elif operation == "Prewitt Operator":

        prewitt_x_kernel = np.array([
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]
        ])

        prewitt_y_kernel = np.array([
            [-1, -1, -1],
            [0, 0, 0],
            [1, 1, 1]
        ])

        prewitt_x = cv2.filter2D(
            gray,
            cv2.CV_64F,
            prewitt_x_kernel
        )

        prewitt_y = cv2.filter2D(
            gray,
            cv2.CV_64F,
            prewitt_y_kernel
        )

        magnitude = cv2.magnitude(
            np.float32(prewitt_x),
            np.float32(prewitt_y)
        )

        result = cv2.convertScaleAbs(
            magnitude
        )

        kernel = prewitt_x_kernel

    # --------------------------------------------------
    # DISPLAY SELECTED KERNEL
    # --------------------------------------------------

    st.subheader("4. Selected Operator / Kernel")

    if operation == "Sobel Operator":

        st.write("Sobel X Kernel:")

        st.dataframe(
            pd.DataFrame(
                sobel_x_kernel
            ),
            hide_index=True
        )

        st.write("Sobel Y Kernel:")

        st.dataframe(
            pd.DataFrame(
                sobel_y_kernel
            ),
            hide_index=True
        )

    elif operation == "Prewitt Operator":

        st.write("Prewitt X Kernel:")

        st.dataframe(
            pd.DataFrame(
                prewitt_x_kernel
            ),
            hide_index=True
        )

        st.write("Prewitt Y Kernel:")

        st.dataframe(
            pd.DataFrame(
                prewitt_y_kernel
            ),
            hide_index=True
        )

    else:

        st.write(
            f"{operation} Kernel:"
        )

        st.dataframe(
            pd.DataFrame(
                kernel
            ),
            hide_index=True
        )

    # --------------------------------------------------
    # OUTPUT IMAGE
    # --------------------------------------------------

    st.subheader(
        "5. Output After Applying Operator"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            gray,
            caption="Input Grayscale Image",
            use_container_width=True
        )

    with col2:

        st.image(
            result,
            caption=f"{operation} Output",
            use_container_width=True
        )

    # --------------------------------------------------
    # OUTPUT PIXEL VALUES
    # --------------------------------------------------

    st.subheader(
        "6. Output Pixel Values"
    )

    st.write(
        "Pixel values after applying the selected "
        "image processing operation:"
    )

    # Limit displayed matrix to 15 × 15
    output_height = min(
        result.shape[0],
        15
    )

    output_width = min(
        result.shape[1],
        15
    )

    output_pixel_matrix = result[
        :output_height,
        :output_width
    ]

    output_pixel_df = pd.DataFrame(
        output_pixel_matrix
    )

    st.write(
        f"Showing first {output_height} × "
        f"{output_width} output pixels:"
    )

    st.dataframe(
        output_pixel_df,
        use_container_width=True
    )

    # --------------------------------------------------
    # OUTPUT PIXEL INFORMATION
    # --------------------------------------------------

    st.write("### Output Pixel Information")

    min_value = int(result.min())
    max_value = int(result.max())
    mean_value = float(result.mean())

    info_col1, info_col2, info_col3 = st.columns(3)

    with info_col1:

        st.metric(
            "Minimum Pixel Value",
            min_value
        )

    with info_col2:

        st.metric(
            "Maximum Pixel Value",
            max_value
        )

    with info_col3:

        st.metric(
            "Average Pixel Value",
            round(mean_value, 2)
        )

    # --------------------------------------------------
    # DOWNLOAD PROCESSED IMAGE
    # --------------------------------------------------

    st.subheader(
        "7. Download Processed Image"
    )

    output_image = Image.fromarray(
        result
    )

    buffer = io.BytesIO()

    output_image.save(
        buffer,
        format="PNG"
    )

    st.download_button(
        label="⬇️ Download Processed Image",
        data=buffer.getvalue(),
        file_name="processed_image.png",
        mime="image/png"
    )

else:

    st.info(
        "👆 Upload an image to start "
        "the image processing."
    )