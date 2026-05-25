document.addEventListener("DOMContentLoaded", function () {
  // ==========================================
  // 1. LOGIC CHUYỂN TAB TỰ ĐỘNG
  // ==========================================
  const tabButtons = document.querySelectorAll(".tab-btn");
  const tabPanels = document.querySelectorAll(".tab-panel");

  tabButtons.forEach((button) => {
    button.addEventListener("click", function () {
      const targetTab = this.getAttribute("data-tab");

      // Loại bỏ class active cũ
      tabButtons.forEach((btn) => btn.classList.remove("active"));
      tabPanels.forEach((panel) => panel.classList.remove("active"));

      // Kích hoạt class active mới
      this.classList.add("active");
      document.getElementById(targetTab).classList.add("active");
    });
  });

  // ==========================================
  // 2. LOGIC PREVIEW ẢNH KHI NGƯỜI DÙNG CHỌN FILE
  // ==========================================
  const singleFileInput = document.getElementById("single-file");
  const compareFileInput = document.getElementById("compare-file");

  singleFileInput.addEventListener("change", function () {
    handleFilePreview(this, "single-view");
  });

  compareFileInput.addEventListener("change", function () {
    // Đối với tab đối chứng, nhân bản ảnh hiển thị ở cả 2 ô của 2 mô hình
    handleFilePreview(this, "compare-view-cnn");
    handleFilePreview(this, "compare-view-rf");
  });

  function handleFilePreview(input, viewId) {
    const viewContainer = document.getElementById(viewId);
    if (input.files && input.files[0]) {
      const reader = new FileReader();
      reader.onload = function (e) {
        viewContainer.innerHTML = `<img src="${e.target.result}" alt="Preview Image">`;
      };
      reader.readAsDataURL(input.files[0]);
    }
  }

  // ==========================================
  // 3. GỌI API BACKEND - NHẬN DIỆN ĐƠN LẺ
  // ==========================================
  const btnPredictSingle = document.getElementById("btn-predict-single");

  btnPredictSingle.addEventListener("click", function () {
    if (!singleFileInput.files[0]) {
      alert("Vui lòng nạp file ảnh hoa lan lên hệ thống trước!");
      return;
    }

    const modelSelect = document.getElementById("model-select").value;
    const resultsWrapper = document.getElementById("single-results");
    const resClass = document.getElementById("res-class");

    // Giao diện chuyển sang trạng thái xử lý tính toán
    resClass.innerText = "Đang trích xuất...";
    resultsWrapper.classList.remove("hidden");

    const formData = new FormData();
    formData.append("file", singleFileInput.files[0]);
    formData.append("model", modelSelect);

    // Gọi API bất đồng bộ đến endpoint của Flask
    fetch("/predict_single", {
      method: "POST",
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.error) {
          alert(data.error);
          return;
        }
        // Điền dữ liệu trả về từ Server vào giao diện
        resClass.innerText = data.class_name;
        document.getElementById("res-conf").innerText = data.confidence;
        document.getElementById("res-time").innerText = data.time;
      })
      .catch((err) => {
        alert("Lỗi kết nối đến Server máy chủ Backend!");
        console.error(err);
      });
  });

  // ==========================================
  // 4. GỌI API BACKEND - ĐỐI CHỨNG SONG SONG
  // ==========================================
  const btnCompare = document.getElementById("btn-compare");

  btnCompare.addEventListener("click", function () {
    if (!compareFileInput.files[0]) {
      alert("Vui lòng nạp file ảnh hoa lan lên hệ thống trước!");
      return;
    }

    const cnnBox = document.getElementById("comp-cnn-box");
    const rfBox = document.getElementById("comp-rf-box");
    const cnnName = document.getElementById("comp-cnn-name");
    const rfName = document.getElementById("comp-rf-name");

    cnnName.innerText = "Đang chạy mạng CNN...";
    rfName.innerText = "Đang trích xuất GLCM/HSV...";

    cnnBox.classList.remove("hidden");
    rfBox.classList.remove("hidden");

    const formData = new FormData();
    formData.append("file", compareFileInput.files[0]);

    fetch("/predict_compare", {
      method: "POST",
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.error) {
          alert(data.error);
          return;
        }
        // Cập nhật nhánh mô hình CNN
        cnnName.innerText = data.cnn.class_name;
        document.getElementById("comp-cnn-conf").innerText =
          data.cnn.confidence;

        // Cập nhật nhánh mô hình Random Forest
        rfName.innerText = data.rf.class_name;
        document.getElementById("comp-rf-conf").innerText = data.rf.confidence;
      })
      .catch((err) => {
        alert("Lỗi hệ thống khi phân tách song song!");
        console.error(err);
      });
  });
});
