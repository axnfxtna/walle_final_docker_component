# walle_final_docker_component

```bash
cd /home/sarucha3/walle_capstone/final_docker_component

./setup.sh

sudo docker compose up -d etcd minio milvus mysql ollama
sudo docker compose build --no-cache walle-data
sudo docker compose up walle-data
```

## Database Overview

### Milvus Collections

| Collection | Source Dataset | Description |
|---|---|---|
| `curriculum` | `dataset/curriculum/*.pdf` | 3 RAI programme course-syllabus PDFs chunked into 2000-char overlapping segments. Fields: `doc_name`, `text_content`, `embedding` (1024-dim). |
| `uni_info` | `dataset/uni_info/*.jpg` + `*.docx` | Campus zone map images (512-dim image vectors) and building location documents (1024-dim text vectors). Fields: `image_embedding`, `doc_embedding`, `file_path`, `file_type`, `text_content`. |
| `time_table` | `dataset/time_table/*.xlsx` | Class and exam schedules parsed into structured Thai sentences (e.g. `"ตาราง RAI 1 รุ่น 67 วันจันทร์ เวลา … วิชา …"`). Primary key = MySQL `ExcelTimetableData.row_id`. Fields: `embedding` (1024-dim). |
| `chat_history` | `database/chat_history/chat_*.json` (runtime) | Every conversation turn stored for long-term memory. `id` = SHA1(person + question). Fields: `person`, `timestamp`, `question`, `answer`, `embedding` (1024-dim). |
| `local_info` | `dataset/bars.json` + `dataset/restaurents.json` | Nearby bars (uses pre-built `text` field) and restaurants (text constructed from name/location/type/review/must_try/price_range). Fields: `source_file`, `name`, `text_content`, `embedding` (1024-dim). |
| `student_manual` | `dataset/student_manual_2564.pdf` | Student handbook PDF chunked into 2000-char overlapping segments. Fields: `doc_name`, `text_content`, `embedding` (1024-dim). |

All collections use **BAAI/bge-m3** (1024-dim) text embeddings, IVF_FLAT index, COSINE metric.

---

### MySQL Tables (database: `capstone`)

| Table | Source Dataset | Description |
|---|---|---|
| `Academic_Year` | `dataset/initial_data.py` → `ACADEMIC_YEARS` | RAI cohort info: `RAI_Gen`, `KMITL_Gen`, `year_start`, `year_end`, `F2D_student_id`. |
| `Students` | `dataset/initial_data.py` → `STUDENTS` | Student profiles: `student_id`, `first_name`, `last_name`, `nick_name`, `student_email`, `enrollment_year`. |
| `Face_Recognition_Data` | `dataset/student_face_image/*.jpg` | Face image paths and encodings per student: `face_id`, `student_id`, `face_image_path`, `face_encoding`. |
| `ExcelTimetableData` | `dataset/time_table/*.xlsx` | Raw timetable sentences mirroring Milvus `time_table` collection. `row_id` (auto-increment) = Milvus primary key for JOIN lookups. |
# walle_final_docker_component
