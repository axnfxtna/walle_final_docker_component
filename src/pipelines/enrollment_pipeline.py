"""
Student Enrollment Pipeline

Interactive pipeline for enrolling students with multi-angle face capture.
Captures 5 angles: straight, left, right, up, down.
Saves to JSON database + Milvus vector store.
"""

import os
import cv2
import numpy as np
from pathlib import Path

from src.utils_enrollment import (
    load_config,
    get_or_create_face_collection,
    insert_face_embeddings,
)
from src.vision.detector_factory import create_scrfd_detector
from src.vision.recognizer import FaceRecognizer
from src.vision.enrollment import EnrollmentManager
from src.vision.database import EnrollmentDatabase


class EnrollmentPipeline:
    """Pipeline for interactive student face enrollment."""

    # ── Angles to capture ────────────────────────────────────────────────────
    ANGLES = [
        ("straight", "Look straight at the camera"),
        ("left",     "Turn your head LEFT (your right)"),
        ("right",    "Turn your head RIGHT (your left)"),
        ("up",       "Tilt your head UP (look at ceiling)"),
        ("down",     "Tilt your head DOWN (look at floor)"),
    ]

    def __init__(self, cfg=None):
        """
        Initialize the enrollment pipeline.

        Args:
            cfg: OmegaConf config object (loaded from settings.yaml).
                 If None, load_config() is called automatically.
        """
        if cfg is None:
            cfg = load_config()
        self.cfg = cfg

        print("=" * 60)
        print("STUDENT ENROLLMENT SYSTEM")
        print("=" * 60)
        print("\nInitializing components...")

        # ── Face detector ────────────────────────────────────────────────────
        print("  Loading face detector...")
        self.detector = create_scrfd_detector(
            model_path=cfg.vision.detector_model,
            confidence_threshold=cfg.vision.get("confidence_threshold", 0.5),
            nms_threshold=0.4,
            input_size=(640, 640),
            device="cuda",
        )

        # ── Face recognizer ──────────────────────────────────────────────────
        print("  Loading face recognizer...")
        self.recognizer = FaceRecognizer(
            model_path=cfg.vision.recognizer_model,
            device="cuda",
        )

        # ── Enrollment quality manager ───────────────────────────────────────
        print("  Loading enrollment manager...")
        self.enrollment_mgr = EnrollmentManager(
            min_face_size=112,
            max_blur_threshold=100.0,
            required_angles=["straight", "left", "right", "up", "down"],
            quality_threshold=0.7,
        )

        # ── JSON database ────────────────────────────────────────────────────
        print("  Loading enrollment database...")
        # In Docker: /app/dataset/registration.json (bind-mounted to host)
        # Locally:   <project_root>/dataset/registration.json
        _project_root = Path(__file__).resolve().parent.parent.parent
        _registration_path = _project_root / "dataset" / "registration.json"
        self.db = EnrollmentDatabase(str(_registration_path))

        # ── Milvus collection ────────────────────────────────────────────────
        print("  Connecting to Milvus...")
        self.milvus_collection = get_or_create_face_collection()

        print(f"\n✓ Initialization complete!")
        print(f"  Current enrollments: {len(self.db)}")
        print()

    # ══════════════════════════════════════════════════════════════════════════
    #  CAPTURE
    # ══════════════════════════════════════════════════════════════════════════

    def capture_angle(self, cap, angle_name: str, angle_instruction: str):
        """
        Capture face from a specific angle via webcam.

        Args:
            cap: OpenCV VideoCapture object.
            angle_name: e.g. "straight", "left".
            angle_instruction: Text shown to the user.

        Returns:
            (embedding, frame) on success, (None, None) on cancel / failure.
        """
        print(f"\n📸 Capturing: {angle_name.upper()}")
        print(f"   {angle_instruction}")
        print("   Press SPACE when ready, ESC to cancel")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("   ❌ Failed to read frame")
                return None, None

            detections = self.detector.detect(frame)

            # ── Draw UI overlay ──────────────────────────────────────────────
            display = frame.copy()
            h, w = display.shape[:2]

            cv2.rectangle(display, (0, 0), (w, 100), (0, 0, 0), -1)
            cv2.putText(display, f"Angle: {angle_name.upper()}", (10, 30),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(display, angle_instruction, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(display, "SPACE = Capture | ESC = Cancel", (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            if len(detections) == 0:
                cv2.putText(display, "NO FACE DETECTED", (w // 2 - 150, h // 2),
                            cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)
            elif len(detections) > 1:
                cv2.putText(display, "MULTIPLE FACES - Please be alone",
                            (w // 2 - 250, h // 2),
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 165, 255), 2)
            else:
                det = detections[0]
                bbox = det["bbox"]
                landmarks = det.get("landmarks")

                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if landmarks:
                    for lx, ly in landmarks:
                        cv2.circle(display, (int(lx), int(ly)), 3, (0, 0, 255), -1)

                quality = self.enrollment_mgr.check_face_quality(
                    frame, bbox, landmarks, expected_angle=angle_name
                )

                feedback_y = y2 + 30
                if quality["passed"]:
                    cv2.putText(display, "✓ GOOD QUALITY - Press SPACE",
                                (x1, feedback_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(display, f"✗ {quality['feedback']}",
                                (x1, feedback_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                metrics_y = feedback_y + 25
                cv2.putText(display, f"Quality: {quality['quality_score']:.2f}",
                            (x1, metrics_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            cv2.imshow("Enrollment - Capture Face", display)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                print("   ⚠️  Cancelled")
                return None, None

            if key == 32:  # SPACE
                if len(detections) == 1:
                    det = detections[0]
                    quality = self.enrollment_mgr.check_face_quality(
                        frame, det["bbox"], det.get("landmarks"),
                        expected_angle=angle_name,
                    )
                    if quality["passed"]:
                        try:
                            embedding = self.recognizer.extract_embedding(
                                frame, det["bbox"], det.get("landmarks")
                            )
                            print(f"   ✓ Captured {angle_name} successfully!")
                            return embedding, frame
                        except Exception as e:
                            print(f"   ❌ Failed to extract embedding: {e}")
                    else:
                        print(f"   ⚠️  Quality check failed: {quality['feedback']}")
                else:
                    print("   ⚠️  Please ensure exactly one face is visible")

    # ══════════════════════════════════════════════════════════════════════════
    #  ENROLL
    # ══════════════════════════════════════════════════════════════════════════

    def enroll_student(self):
        """Run the full enrollment process for a new student."""
        print("\n" + "=" * 60)
        print("NEW STUDENT ENROLLMENT")
        print("=" * 60)

        # Use network camera stream if CAMERA_URL is set, otherwise local webcam
        camera_source = os.getenv("CAMERA_URL", "0")
        if camera_source == "0":
            camera_source = 0  # local device index
        print(f"📷 Camera source: {camera_source or 'local webcam'}")

        cap = cv2.VideoCapture(camera_source)
        if not cap.isOpened():
            print("❌ Failed to open camera")
            print("   Tip: Set CAMERA_URL env var for network camera")
            print("   Example: CAMERA_URL=http://<laptop-ip>:8080/video")
            return

        print("\nStarting face capture...")
        print("Follow the on-screen instructions for each angle")

        embeddings = []
        captured_images = []

        for angle_name, instruction in self.ANGLES:
            embedding, image = self.capture_angle(cap, angle_name, instruction)
            if embedding is None:
                print("\n⚠️  Enrollment cancelled")
                cap.release()
                cv2.destroyAllWindows()
                return
            embeddings.append(embedding)
            captured_images.append(image)

        cap.release()
        cv2.destroyAllWindows()

        # ── Validate embeddings ──────────────────────────────────────────────
        print("\n🔍 Validating embeddings...")
        if not self.enrollment_mgr.validate_embeddings(embeddings):
            print("❌ Embedding validation failed - please try again")
            return
        print("✓ Embeddings validated successfully!")

        # ── Collect student info ─────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("STUDENT INFORMATION")
        print("=" * 60)

        student_id = input("\nEnter Student ID: ").strip()
        if not student_id:
            print("❌ Student ID cannot be empty")
            return

        if self.db.get_student(student_id):
            overwrite = input(
                f"⚠️  Student {student_id} already exists. Overwrite? (y/n): "
            ).strip().lower()
            if overwrite != "y":
                print("❌ Enrollment cancelled")
                return

        student_fullname_thai = input("Enter Student Thai Full Name: ").strip()
        if not student_fullname_thai:
            print("❌ Student name cannot be empty")
            return

        student_fullname_eng = input("Enter Student English Full Name: ").strip()
        if not student_fullname_eng:
            print("❌ Student name cannot be empty")
            return

        student_nickname_thai = input("Enter Student Thai Nickname: ").strip()
        if not student_nickname_thai:
            print("❌ Student name cannot be empty")
            return

        student_nickname_eng = input("Enter Student English Nickname: ").strip()
        if not student_nickname_eng:
            print("❌ Student name cannot be empty")
            return

        gen = input("Enter RAI Gen (optional): ").strip()
        section = input("Enter Section (optional): ").strip()

        metadata = {}
        if gen:
            metadata["RAI Gen"] = gen
        if section:
            metadata["Section"] = section

        # ── Save ─────────────────────────────────────────────────────────────
        print("\n💾 Saving enrollment...")
        success = self.db.enroll_student(
            student_id=student_id,
            fullname_thai=student_fullname_thai,
            fullname_eng=student_fullname_eng,
            nickname_thai=student_nickname_thai,
            nickname_eng=student_nickname_eng,
            embeddings=embeddings,
            metadata=metadata,
        )

        if success:
            print("  Inserting embeddings into Milvus...")
            insert_face_embeddings(self.milvus_collection, student_id, embeddings)

            print("\n" + "=" * 60)
            print("✅ ENROLLMENT SUCCESSFUL!")
            print("=" * 60)
            print(f"Student ID: {student_id}")
            print(f"Name: {student_fullname_thai}")
            print(f"Embeddings captured: {len(embeddings)}")
            print(f"Saved to: registration.json + Milvus 'student_face_images'")
            print(f"Total enrolled students: {len(self.db)}")
            print("=" * 60)
        else:
            print("\n❌ Failed to save enrollment")

    # ══════════════════════════════════════════════════════════════════════════
    #  LIST
    # ══════════════════════════════════════════════════════════════════════════

    def list_students(self):
        """List all enrolled students."""
        students = self.db.get_all_students()

        if not students:
            print("\n📋 No students enrolled yet")
            return

        print("\n" + "=" * 60)
        print("ENROLLED STUDENTS")
        print("=" * 60)

        for student_id, info in students.items():
            print(f"\nID: {student_id}")
            print(f"  Name: {info['name']}")
            print(f"  Enrolled: {info['enrolled_date']}")
            print(f"  Embeddings: {len(info['embeddings'])}")
            if info.get("metadata"):
                print(f"  Metadata: {info['metadata']}")

        print("=" * 60)

    # ══════════════════════════════════════════════════════════════════════════
    #  RUN (interactive menu)
    # ══════════════════════════════════════════════════════════════════════════

    def run(self):
        """Run the interactive enrollment menu."""
        while True:
            print("\n" + "=" * 60)
            print("ENROLLMENT MENU")
            print("=" * 60)
            print("1. Enroll new student")
            print("2. List enrolled students")
            print("3. Exit")
            print("=" * 60)

            choice = input("\nEnter choice (1-3): ").strip()

            if choice == "1":
                self.enroll_student()
            elif choice == "2":
                self.list_students()
            elif choice == "3":
                print("\n👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice")
