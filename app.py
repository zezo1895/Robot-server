# server.py - CogniCar v20 COMPLETE - مع نظام تحكم كامل في السيارة
import os
import json
import urllib.request
import face_recognition
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import io
from PIL import Image
import firebase_admin
from firebase_admin import credentials, firestore
import cloudinary
import cloudinary.uploader
import uuid
import time
import threading
import urllib.parse

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ==================== تحميل الموديلات تلقائيًا ====================
def download_models():
    """تحميل الموديلات من Google Drive إذا لم تكن موجودة"""
    if not os.path.exists("deploy.prototxt"):
        print("جاري تحميل deploy.prototxt...")
        urllib.request.urlretrieve(
            "https://drive.google.com/uc?export=download&id=1jz2DuFSpPXNlPsE_5N3URzaciQOwaeSO",
            "deploy.prototxt"
        )
        print("✅ تم تحميل deploy.prototxt")
    
    if not os.path.exists("res10_300x300_ssd_iter_140000.caffemodel"):
        print("جاري تحميل موديل الوجه (23 ميجا)...")
        urllib.request.urlretrieve(
            "https://drive.google.com/uc?export=download&id=1_KoGu_MY47gZJ4sVlF1bZZnRbpkJ2dbn",
            "res10_300x300_ssd_iter_140000.caffemodel"
        )
        print("✅ تم تحميل الموديل الكبير")

# شغّل التحميل أول ما السيرفر يبدأ
download_models()

# ==================== Firebase ====================
try:
    firebase_config = json.loads(os.environ['FIREBASE_CONFIG'])
    cred = credentials.Certificate(firebase_config)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("[✅] Firebase متصل")
except Exception as e:
    db = None
    print(f"[⚠️] Firebase: {e}")

# ==================== Cloudinary ====================
cloudinary.config(
    cloud_name=os.environ['CLOUD_NAME'],
    api_key=os.environ['CLOUD_API_KEY'],
    api_secret=os.environ['CLOUD_API_SECRET']
)

# ==================== نظام التحكم بالسيارة ====================
current_command = "auto"  # 🎮 الأمر الحالي للسيارة
command_timestamp = time.time()  # ⏰ وقت آخر أمر
command_history = []  # 📜 سجل الأوامر

# 🎯 قائمة الأوامر الصالحة
VALID_COMMANDS = [
    'forward', 'backward', 'left', 'right', 'stop', 'auto',
    'forward_pulse', 'backward_pulse', 'left_pulse', 'right_pulse'
]

# ==================== رفع الصور ====================
def upload_async(data, public_id, format="jpg"):
    """📤 رفع الصور لـ Cloudinary في خيط منفصل"""
    def task():
        try:
            cloudinary.uploader.upload(
                io.BytesIO(data),
                public_id=public_id,
                folder="faces",
                overwrite=True,
                use_filename=True,
                unique_filename=False,
                format=format,
                timeout=90
            )
            print(f"[📤 UPLOADED] faces/{public_id}.{format}")
        except Exception as e:
            print(f"[❌ ERROR] رفع {public_id}: {e}")
    threading.Thread(target=task, daemon=True).start()

# ==================== حذف الصور من Cloudinary ====================
def delete_from_cloudinary_async(doc_id):
    """🗑️ حذف الصور من Cloudinary (واحدة واحدة – مضمون 100%)"""
    def task():
        poses = ['front', 'left', 'right']
        for pose in poses:
            public_id = f"faces/{doc_id}_{pose}"
            try:
                result = cloudinary.uploader.destroy(public_id, invalidate=True)
                status = "✅ حذف" if result.get('result') == 'ok' else "ℹ️ غير موجود"
                print(f"[☁️ CLOUDINARY] {public_id} → {status}")
            except Exception as e:
                print(f"[❌ ERROR] حذف {public_id}: {e}")
    threading.Thread(target=task, daemon=True).start()

# ==================== التعرف على الوجوه ====================
known_faces = {}  # 💾 الوجوه المخزنة محلياً
last_update = 0  # ⏰ آخر تحديث للذاكرة

def load_all_faces():
    """🔄 تحميل جميع الوجوه من Firebase"""
    global known_faces, last_update
    if time.time() - last_update < 3 or not db:
        return
    temp = {}
    try:
        for doc in db.collection('known_faces').stream():
            data = doc.to_dict()
            name = data.get('name')
            if not name or 'encoding_front' not in data:
                continue
            front = np.array(data['encoding_front'])
            left = np.array(data.get('encoding_left', front))
            right = np.array(data.get('encoding_right', front))
            temp[name] = [front, left, right]
        known_faces = temp
        last_update = time.time()
        print(f"[👥 FACES] تم تحميل {len(known_faces)} وجه")
    except Exception as e:
        print(f"[⚠️] خطأ في تحميل الوجوه: {e}")
load_all_faces()

def get_single_encoding(image_bytes):
    """🧠 استخراج ترميز وجه من الصورة"""
    try:
        # 📉 تصغير الصورة الكبيرة
        if len(image_bytes) > 800_000:
            img = Image.open(io.BytesIO(image_bytes)).resize((300, 300))
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=70)
            image_bytes = buf.getvalue()
        
        img = face_recognition.load_image_file(io.BytesIO(image_bytes))
        locations = face_recognition.face_locations(img, number_of_times_to_upsample=0, model="hog")
        
        if not locations:
            return None
        
        encodings = face_recognition.face_encodings(img, locations, num_jitters=1)
        
        if len(encodings) > 1:
            # 📏 اختيار أكبر وجه
            areas = [(b-t)*(r-l) for t,r,b,l in locations]
            return encodings[areas.index(max(areas))]
        
        return encodings[0]
    except Exception as e:
        print(f"[❌] خطأ في استخراج الترميز: {e}")
        return None

# ==================== API Routes ====================

# 🏥 فحص حالة السيرفر
@app.route('/health', methods=['GET'])
def health_check():
    """🏥 فحص حالة السيرڤر وإعادة تحميل الوجوه"""
    load_all_faces()
    return jsonify({
        "status": "👍 OK",
        "faces_count": len(known_faces),
        "current_command": current_command,
        "uptime": round(time.time() - command_timestamp, 1)
    })

# 🎮 تعيين أمر للسيارة
@app.route('/set_command', methods=['POST'])
def set_command():
    """🎮 تعيين أمر جديد للتحكم في السيارة"""
    global current_command, command_timestamp, command_history
    
    try:
        data = request.json
        command = data.get('command')
        
        if command not in VALID_COMMANDS:
            print(f"[❌] أمر غير صالح: {command}")
            return jsonify({"error": "🚫 أمر غير صالح"}), 400
        
        # تحديث الأمر
        old_command = current_command
        current_command = command
        command_timestamp = time.time()
        
        # حفظ في السجل
        command_history.append({
            "command": command,
            "timestamp": time.time(),
            "old_command": old_command
        })
        
        # 🔄 تقليل حجم السجل
        if len(command_history) > 100:
            command_history.pop(0)
        
        print(f"[🎮 COMMAND] {old_command} → {command}")
        return jsonify({
            "status": "✅ OK",
            "command": command,
            "timestamp": command_timestamp
        })
    except Exception as e:
        print(f"[❌] خطأ في تعيين الأمر: {e}")
        return jsonify({"error": "❌ فشل تعيين الأمر"}), 500

# 🎮 الحصول على الأمر الحالي
@app.route('/get_command', methods=['GET'])
def get_command():
    """🎮 الحصول على الأمر الحالي للسيارة"""
    try:
        return jsonify({
            "status": "✅ OK",
            "command": current_command,
            "timestamp": command_timestamp,
            "age": round(time.time() - command_timestamp, 2)
        })
    except Exception as e:
        print(f"[❌] خطأ في جلب الأمر: {e}")
        return jsonify({"error": "❌ فشل جلب الأمر"}), 500

# 📡 تأكيد استلام الأمر من ESP
@app.route('/esp_ack', methods=['POST'])
def esp_ack():
    """📡 استلام تأكيد من شريحة ESP-01 على استلام الأمر"""
    try:
        data = request.json
        command = data.get('command')
        print(f"\n[📡 ESP-01 ACK] ✅ تم استلام الأمر: {command}\n")  
        return jsonify({"status": "✅ ACK_RECEIVED"})
    except Exception as e:
        print(f"[❌] خطأ في ACK: {e}")
        return jsonify({"error": "❌ فشل معالجة ACK"}), 400

# 👤 تسجيل وجه جديد
@app.route('/register', methods=['POST'])
def register():
    """👤 تسجيل وجه جديد بثلاث وضعيات"""
    name = request.form.get('name', '').strip()
    if not name or len(name.split()) < 2:
        return jsonify({"error": "❌ الاسم يجب أن يكون اسمين"}), 400

    encodings = {}
    images = {}
    doc_id = str(uuid.uuid4())

    for pose in ['front', 'left', 'right']:
        if pose not in request.files:
            return jsonify({"error": f"❌ مفقود {pose}"}), 400
        file = request.files[pose]
        image_data = file.read()
        enc = get_single_encoding(image_data)
        if enc is None:
            return jsonify({"error": f"❌ لا يوجد وجه في {pose}"}), 400
        encodings[pose] = enc.tolist()
        images[pose] = image_data

    load_all_faces()
    test_enc = np.array(encodings['front'])
    for known_name, encs in known_faces.items():
        if min(face_recognition.face_distance(encs, test_enc)) < 0.5:
            return jsonify({"error": f"❌ موجود بالفعل: {known_name}"}), 400

    for pose in ['front', 'left', 'right']:
        upload_async(images[pose], f"{doc_id}_{pose}", "jpg")

    base_url = "https://res.cloudinary.com/dab3zstzc/image/upload/faces"
    if db:
        db.collection('known_faces').document(doc_id).set({
            'name': name,
            'encoding_front': encodings['front'],
            'encoding_left': encodings['left'],
            'encoding_right': encodings['right'],
            'image_front': f"{base_url}/{doc_id}_front.jpg",
            'image_left': f"{base_url}/{doc_id}_left.jpg",
            'image_right': f"{base_url}/{doc_id}_right.jpg",
            'timestamp': firestore.SERVER_TIMESTAMP
        })

    known_faces[name] = [
        np.array(encodings['front']), 
        np.array(encodings['left']), 
        np.array(encodings['right'])
    ]
    
    # 🚗 إذا تم تسجيل شخص جديد، انتقل للوضع التلقائي
    global current_command
    current_command = "auto"
    
    print(f"[✅ REGISTERED] {name} تم التسجيل بنجاح! 🎉")
    return jsonify({
        "status": "✅ REGISTERED", 
        "name": name,
        "message": f"🎉 تم تسجيل {name} بنجاح!"
    })

# 🧠 التعرف على وجه
@app.route('/recognize', methods=['POST'])
def recognize():
    """🧠 التعرف على وجه من الصورة"""
    if 'image' not in request.files:
        return jsonify({"status": "❌ ERROR"}), 400
    
    load_all_faces()
    encoding = get_single_encoding(request.files['image'].read())
    
    if encoding is None:
        return jsonify({"status": "❌ NO_FACE"}), 400
    
    if not known_faces:
        return jsonify({"status": "❓ UNKNOWN"}), 200
    
    distances = {n: min(face_recognition.face_distance(e, encoding)) for n, e in known_faces.items()}
    best_name = min(distances, key=distances.get)
    best_dist = distances[best_name]
    
    if best_dist < 0.52:
        confidence = round((1 - best_dist) * 100, 1)
        
        # 🚗 تغيير أمر السيارة بناءً على الشخص المعروف
        if best_name in ["محمد أحمد", "علي حسين"]:
            global current_command
            current_command = "forward"
            print(f"[🎮] تحويل الأمر إلى forward للشخص: {best_name}")
        
        return jsonify({
            "status": "✅ MATCHED",
            "name": best_name,
            "confidence": f"{confidence}%",
            "message": f"🎉 أهلاً {best_name}!",
            "command": current_command  # إرجاع الأمر الحالي
        })
    
    return jsonify({"status": "❓ UNKNOWN"})

# 📋 قائمة الأشخاص المسجلين
@app.route('/list_people', methods=['GET'])
def list_people():
    """📋 الحصول على قائمة جميع الأشخاص المسجلين"""
    result = []
    try:
        if db:
            for doc in db.collection('known_faces').stream():
                data = doc.to_dict()
                name = data.get('name', 'غير معروف')
                front = data.get('image_front') or f"https://via.placeholder.com/150/007AFF/FFFFFF?text={name.split()[0]}"
                result.append({
                    "id": doc.id,
                    "name": name,
                    "image_front": front,
                    "image_left": data.get('image_left', ''),
                    "image_right": data.get('image_right', '')
                })
    except Exception as e:
        print(f"[❌ ERROR] list_people: {e}")
    
    return jsonify(result)

# 🗑️ حذف شخص مسجل
@app.route('/delete/<name>', methods=['DELETE'])
def delete_person(name):
    """🗑️ حذف شخص مسجل من النظام"""
    try:
        name = urllib.parse.unquote(name)
        docs = list(db.collection('known_faces').where('name', '==', name).stream()) if db else []

        for doc in docs:
            doc_id = doc.id
            # حذف من Firebase
            doc.reference.delete()
            # حذف الصور من Cloudinary
            delete_from_cloudinary_async(doc_id)

        known_faces.pop(name, None)
        print(f"[🗑️ DELETED] {name} + صوره اتحذفت من Cloudinary")
        return jsonify({"status": "✅ deleted"})
    except Exception as e:
        print(f"[❌ ERROR] حذف: {e}")
        return jsonify({"error": "❌ فشل الحذف"}), 500

# 📜 سجل الأوامر
@app.route('/command_history', methods=['GET'])
def get_command_history():
    """📜 الحصول على سجل الأوامر الأخيرة"""
    return jsonify({
        "status": "✅ OK",
        "current": current_command,
        "history": command_history[-20:],  # آخر 20 أمر
        "count": len(command_history)
    })

# 🔄 إعادة تعيين السيارة
@app.route('/reset_car', methods=['POST'])
def reset_car():
    """🔄 إعادة تعيين السيارة للوضع الافتراضي"""
    global current_command, command_timestamp
    current_command = "auto"
    command_timestamp = time.time()
    print(f"[🔄 RESET] السيارة عادت للوضع: auto")
    return jsonify({
        "status": "✅ RESET",
        "command": "auto",
        "message": "🔄 تم إعادة تعيين السيارة"
    })

# ==================== تشغيل السيرڤر ====================
if __name__ == '__main__':
    print("="*80)
    print("🚗 CogniCar v20 COMPLETE - النظام الكامل")
    print("🎮 نظام تحكم السيارة + تعرف على الوجوه")
    print("="*80)
    print("📌 Endpoints:")
    print("🏥  GET  /health                    - فحص حالة السيرڤر")
    print("🎮  POST /set_command               - تعيين أمر للسيارة")
    print("🎮  GET  /get_command               - جلب الأمر الحالي")
    print("📡  POST /esp_ack                    - تأكيد من ESP")
    print("👤  POST /register                   - تسجيل وجه جديد")
    print("🧠  POST /recognize                  - التعرف على وجه")
    print("📋  GET  /list_people               - قائمة الأشخاص")
    print("🗑️  DELETE /delete/<name>           - حذف شخص")
    print("📜  GET  /command_history           - سجل الأوامر")
    print("🔄  POST /reset_car                  - إعادة تعيين السيارة")
    print("="*80)
    
    port = int(os.environ.get('PORT', 5000))
    try:
        from waitress import serve
        serve(app, host="0.0.0.0", port=port, threads=16)
    except ImportError:
        print("[⚠️] Waitress غير مثبت، استخدام Flask development server")
        app.run(host="0.0.0.0", port=port, threaded=True, debug=False)
