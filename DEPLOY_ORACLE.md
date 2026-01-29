# מדריך העלאת בוט ל-Oracle Cloud (חינם לתמיד)

## שלב 1: יצירת חשבון Oracle Cloud

1. גש ל-https://www.oracle.com/cloud/free/
2. לחץ על "Start for free"
3. מלא פרטים:
   - אימייל
   - מדינה: Israel
   - שם מלא
4. אימות טלפון (SMS)
5. **חשוב:** צריך כרטיס אשראי לאימות (לא יחייבו כלום)
6. בחר "Free Tier" (לא "Pay As You Go")

---

## שלב 2: יצירת VM Instance

1. התחבר ל-https://cloud.oracle.com/
2. מתפריט ☰ (למעלה משמאל) → **Compute** → **Instances**
3. לחץ **Create Instance**

### הגדרות VM:
- **Name:** `kraken-bot` (או כל שם)
- **Image:** Ubuntu 22.04 (ברירת מחדל) או Oracle Linux 9
- **Shape:** 
  - לחץ "Change Shape"
  - בחר **VM.Standard.E2.1.Micro** (Always Free)
  - 1 CPU, 1GB RAM
- **Networking:** השאר ברירת מחדל
- **Add SSH Keys:**
  - בחר "Generate SSH key pair"
  - לחץ **Save Private Key** → שמור בשם `oracle_key.pem` במחשב שלך
  - לחץ **Save Public Key** (גם כן שמור)

4. לחץ **Create** ➜ המתן 1-2 דקות

הערה חשובה:
- אם בחרת Oracle Linux, שם המשתמש להתחברות ב-SSH הוא `opc` (ולא `ubuntu`).
- אם בחרת Ubuntu, שם המשתמש הוא `ubuntu`.

---

## שלב 3: פתיחת פורטים (Firewall)

1. לאחר יצירת ה-VM, היכנס ל-**Instance Details**
2. תחת **Primary VNIC** → לחץ על ה-Subnet
3. תחת **Security Lists** → לחץ על ה-Default Security List
4. לחץ **Add Ingress Rules**:
   - **Source CIDR:** `0.0.0.0/0`
   - **Destination Port:** `22` (SSH)
   - לחץ **Add Ingress Rules**

*(אופציונלי: אם תרצה Web UI בעתיד, פתח גם פורט 8080 או 5000)*

הקצאת Public IP (חינמי):
- ב-Oracle Cloud ניתן להקצות **Ephemeral Public IP** ללא עלות במסגרת ה-Free Tier. אין צורך ב־Reserved Public IP עבור שרת בדיקה.
- כדי שהאפשרות תופיע, ה־Instance חייב להיות בסאב־נט מסוג **Public** שמאפשר הקצאת Public IP.

אם אינך רואה כפתור Assign public IP:
1. עבור ל־Networking → Virtual Cloud Networks → בחר את ה־VCN שלך.
2. לחץ Subnets → **Create Subnet**:
  - בחר **Public Subnet**.
  - סמן **Assign a public IPv4 address** / **Allow public IP address**.
  - שמור.
3. חזור ל־Instance → לשונית **Networking**.
  - אפשרות א': לחץ **Create VNIC** והצמד VNIC חדש לסאב־נט הציבורי, וסמן **Assign ephemeral public IP**.
  - אפשרות ב' (פשוטה לחדשים): צור Instance חדש ובמסך ה־Networking בחר את הסאב־נט הציבורי וסמן **Assign public IPv4 address**.
4. אחרי ההצמדה/יצירה, ודא שב־Primary/Attached VNICs מופיע **Public IPv4 address** ולא מקף.

---

## שלב 4: התחברות לשרת

### ב-macOS/Linux:
```bash
chmod 400 ~/Downloads/oracle_key.pem
## Oracle Linux:
ssh -i ~/Downloads/oracle_key.pem opc@<PUBLIC_IP>

## Ubuntu:
ssh -i ~/Downloads/oracle_key.pem ubuntu@<PUBLIC_IP>
```

**החלף `<PUBLIC_IP>`** עם ה-Public IP Address שמוצג ב-Instance Details.

---

## שלב 5: התקנת הבוט בשרת

לאחר התחברות ל-SSH, הרץ:

### 1. עדכון מערכת והתקנת Python:
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3 python3-pip python3-venv git -y
```

### 2. העלאת הקוד:
**באפשרותך:**

**אופציה A - העתקה ידנית (מהמחשב שלך):**
פתח terminal חדש במחשב המקומי שלך:
```bash
cd /Users/galben/Desktop/Gal/קריפטו/פיתוח/Dev
scp -i ~/Downloads/oracle_key.pem -r . ubuntu@<PUBLIC_IP>:~/kraken-bot/
```

**אופציה B - שימוש ב-Git (מומלץ):**
אם הקוד ב-GitHub:
```bash
# בשרת:
cd ~
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git kraken-bot
cd kraken-bot
```

### 3. הרצת סקריפט ההתקנה:
```bash
cd ~/kraken-bot
chmod +x scripts/setup_server.sh
./scripts/setup_server.sh
```

### 4. הגדרת `.env`:
```bash
nano .env
```
הדבק את התוכן מ-.env המקומי שלך (API_KEY, SECRET, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
שמור: `Ctrl+O`, `Enter`, יציאה: `Ctrl+X`

### 5. בדיקה ידנית:
```bash
source venv/bin/activate
python src/main.py
```
אם הכל עובד, עצור עם `Ctrl+C`

---

## שלב 6: הפיכת הבוט לשירות (רץ תמיד)

### 1. הפעל את סקריפט יצירת השירות:
```bash
sudo ./scripts/create_service.sh
```

### 2. ניהול השירות:
```bash
# התחל את הבוט
sudo systemctl start kraken-bot

# בדוק סטטוס
sudo systemctl status kraken-bot

# לוגים חיים
sudo journalctl -u kraken-bot -f

# עצירה
sudo systemctl stop kraken-bot

# הפעלה מחדש
sudo systemctl restart kraken-bot
```

השירות יתחיל אוטומטית לאחר reboot ויתאושש אוטומטית אם נופל.

---

## שלב 7: ניטור

### צפייה בלוגים:
```bash
# לוגי מערכת
sudo journalctl -u kraken-bot -f

# לוגי טריידים
tail -f ~/kraken-bot/logs/trades.json
```

### בדיקת סטטוס שרת:
```bash
# בדיקת זיכרון ו-CPU
htop

# בדיקת נפח דיסק
df -h
```

---

## פתרון בעיות נפוצות

### הבוט לא מתחיל:
```bash
# בדוק שגיאות
sudo journalctl -u kraken-bot -n 50

# ריצה ידנית לבדיקה
cd ~/kraken-bot
source venv/bin/activate
python src/main.py
```

### שכחתי את ה-IP:
חזור ל-https://cloud.oracle.com/ → Compute → Instances

### חיבור SSH נכשל:
```bash
# וודא הרשאות
chmod 400 ~/Downloads/oracle_key.pem

# נסה עם verbose
ssh -v -i ~/Downloads/oracle_key.pem ubuntu@<PUBLIC_IP>
```

---

## עדכון הבוט בעתיד

```bash
# התחבר לשרת
ssh -i ~/Downloads/oracle_key.pem ubuntu@<PUBLIC_IP>

# עדכן קוד
cd ~/kraken-bot
git pull  # אם משתמש ב-Git
# או העתק קבצים חדשים עם scp

# הפעל מחדש
sudo systemctl restart kraken-bot
```

---

## אבטחה (מומלץ מאוד!)

### 1. שנה סיסמת ubuntu:
```bash
sudo passwd ubuntu
```

### 2. הגבל SSH ל-IP שלך בלבד:
ב-Oracle Cloud Console → Security List → ערוך את Ingress Rule לפורט 22:
- במקום `0.0.0.0/0` שים את ה-IP הציבורי שלך (בדוק ב-https://whatismyip.com)

### 3. הגדר UFW (firewall):
```bash
sudo ufw allow 22/tcp
sudo ufw enable
```

---

## עלות

✅ **0 ש"ח לחודש** (Always Free Tier)
- VM.Standard.E2.1.Micro עד 2 instances
- 1GB RAM, 1 vCPU
- 10TB/חודש bandwidth

---

🎉 **סיימת!** הבוט שלך עכשיו רץ 24/7 על שרת חינמי.
