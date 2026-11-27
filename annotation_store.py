"""Versioned SQLite annotation/draft store; immutable revisions survive reruns."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

STATUSES = {"unlabeled", "proposed", "reviewed", "rejected", "needs_review"}


class AnnotationStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS revisions (
                id INTEGER PRIMARY KEY, image_id TEXT NOT NULL, version INTEGER NOT NULL,
                payload TEXT NOT NULL, saved_at TEXT NOT NULL,
                UNIQUE(image_id,version))""")
            connection.execute("CREATE TABLE IF NOT EXISTS drafts(image_id TEXT PRIMARY KEY,payload TEXT NOT NULL,base_version INTEGER NOT NULL)")
            connection.execute("CREATE TABLE IF NOT EXISTS preferences(key TEXT PRIMARY KEY,value TEXT NOT NULL)")

    def latest(self, image_id):
        with sqlite3.connect(self.path) as connection:
            row = connection.execute("SELECT payload FROM revisions WHERE image_id=? ORDER BY version DESC LIMIT 1", (image_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def save(self, payload, expected_version=None):
        row = json.loads(json.dumps(payload))
        if row.get("review_status") not in STATUSES:
            raise ValueError("Invalid review status")
        if not row.get("image_id") or not row.get("image_path"):
            raise ValueError("image_id and image_path required")
        if row["review_status"] == "reviewed" and (not row.get("source_group") or not row.get("confirmed_by_user")):
            raise ValueError("Reviewed labels need source_group and explicit human confirmation")
        with sqlite3.connect(self.path, timeout=10) as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute("SELECT MAX(version) FROM revisions WHERE image_id=?", (row["image_id"],)).fetchone()[0] or 0
            if expected_version is not None and current != expected_version:
                raise ValueError("Annotation changed in another session; reload before saving")
            row["annotation_version"] = current + 1
            now = datetime.now(timezone.utc).isoformat()
            row["reviewed_at"] = now if row["review_status"] == "reviewed" else None
            connection.execute("INSERT INTO revisions(image_id,version,payload,saved_at) VALUES(?,?,?,?)",
                               (row["image_id"], current+1, json.dumps(row, ensure_ascii=False), now))
        return row

    def history(self, image_id):
        with sqlite3.connect(self.path) as connection:
            return [json.loads(r[0]) for r in connection.execute("SELECT payload FROM revisions WHERE image_id=? ORDER BY version", (image_id,))]

    def migrate_splits(self, changes, expected_rows, operation_id):
        """Atomic approved split-only revisions. Exact preconditions, idempotent.

        Preserve every payload field except split/version; keep reviewed_at as
        the original label-review time. saved_at records migration time. Drafts
        are advanced only when identical to the old reviewed payload; divergent
        drafts retain their base and will conflict rather than lose edits.
        """
        import copy
        if len({c['image_id'] for c in changes})!=len(changes):
            raise ValueError('Duplicate migration identities')
        with sqlite3.connect(self.path,timeout=30) as connection:
            connection.execute('BEGIN IMMEDIATE')
            connection.execute('CREATE TABLE IF NOT EXISTS split_migrations(operation_id TEXT PRIMARY KEY,receipt TEXT NOT NULL)')
            receipt=connection.execute('SELECT receipt FROM split_migrations WHERE operation_id=?',(operation_id,)).fetchone()
            if receipt:return {**json.loads(receipt[0]),'already_applied':True}
            actual={identity:json.loads(payload) for identity,payload in connection.execute(
                'SELECT r.image_id,r.payload FROM revisions r JOIN (SELECT image_id,MAX(version) v FROM revisions GROUP BY image_id)t ON r.image_id=t.image_id AND r.version=t.v')}
            reviewed={k:v for k,v in actual.items() if v.get('review_status')=='reviewed'}
            if reviewed!=expected_rows:
                raise ValueError('SQLite no longer matches approved preflight snapshot')
            saved=[]
            for change in changes:
                identity=change['image_id'];old=actual[identity]
                if old['split']!=change['from'] or old['annotation_version']!=change['base_revision'] or old['source_group']!=change['source_group']:
                    raise ValueError('Migration base revision/group/split mismatch')
                if change['from'] not in ('train','val') or change['proposed_to'] not in ('train','val') or change['from']==change['proposed_to']:
                    raise ValueError('Only explicitly approved train/val migration permitted')
                row=copy.deepcopy(old);row['split']=change['proposed_to'];row['annotation_version']+=1
                now=datetime.now(timezone.utc).isoformat()
                connection.execute('INSERT INTO revisions(image_id,version,payload,saved_at) VALUES(?,?,?,?)',
                                   (identity,row['annotation_version'],json.dumps(row,ensure_ascii=False),now))
                draft=connection.execute('SELECT payload,base_version FROM drafts WHERE image_id=?',(identity,)).fetchone()
                if draft and json.loads(draft[0])==old and draft[1]==old['annotation_version']:
                    connection.execute('UPDATE drafts SET payload=?,base_version=? WHERE image_id=?',
                                       (json.dumps(row,ensure_ascii=False),row['annotation_version'],identity))
                saved.append({'image_id':identity,'before':old,'after':row,'saved_at':now})
            receipt={'operation_id':operation_id,'changes':saved,'already_applied':False}
            connection.execute('INSERT INTO split_migrations VALUES(?,?)',(operation_id,json.dumps(receipt,ensure_ascii=False)))
        return receipt

    def all_latest(self):
        with sqlite3.connect(self.path) as connection:
            return {identity: json.loads(payload) for identity,payload in connection.execute(
                "SELECT r.image_id,r.payload FROM revisions r JOIN (SELECT image_id,MAX(version) v FROM revisions GROUP BY image_id) t ON r.image_id=t.image_id AND r.version=t.v")}

    def draft(self, identity):
        with sqlite3.connect(self.path) as connection:
            row=connection.execute("SELECT payload,base_version FROM drafts WHERE image_id=?",(identity,)).fetchone()
        return (json.loads(row[0]),row[1]) if row else None

    def save_draft(self, payload, base_version):
        """Draft never changes approved revision/export eligibility.

        Compare base revision to prevent stale drafts overwriting newer review.
        A draft is a recoverable working copy, not human ground truth.
        """
        with sqlite3.connect(self.path,timeout=10) as connection:
            connection.execute("BEGIN IMMEDIATE")
            version=connection.execute("SELECT MAX(version) FROM revisions WHERE image_id=?",(payload['image_id'],)).fetchone()[0] or 0
            if version!=base_version:
                raise ValueError("Review changed in another session; reload before autosave")
            connection.execute("INSERT INTO drafts VALUES(?,?,?) ON CONFLICT(image_id) DO UPDATE SET payload=excluded.payload,base_version=excluded.base_version",
                               (payload['image_id'],json.dumps(payload,ensure_ascii=False),base_version))

    def clear_draft(self, identity):
        with sqlite3.connect(self.path) as connection:
            connection.execute("DELETE FROM drafts WHERE image_id=?",(identity,))

    def preference(self,key,default=None):
        with sqlite3.connect(self.path) as connection:
            row=connection.execute("SELECT value FROM preferences WHERE key=?",(key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_preference(self,key,value):
        with sqlite3.connect(self.path) as connection:
            connection.execute("INSERT INTO preferences VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",(key,json.dumps(value)))

    def restore(self, identity, version, expected_version):
        """Restore old contents as a NEW needs-review revision, never approve implicitly."""
        revision=next((row for row in self.history(identity) if row['annotation_version']==version),None)
        if revision is None:
            raise ValueError("Revision not found")
        revision.update(review_status="needs_review",confirmed_by_user=False,restored_from=version)
        return self.save(revision,expected_version)


def image_id(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canvas_boxes(objects, scale, image_width, image_height):
    """Fabric rect scale/position → original pixel xywh; reject out-of-bounds."""
    if scale<=0:
        raise ValueError("Scale must be positive")
    boxes = []
    for obj in objects:
        if obj.get("type") != "rect":
            continue
        if float(obj.get("angle", 0)) != 0:
            raise ValueError("Rotated canvas rectangles unsupported; use axis-aligned boxes")
        x, y = float(obj["left"])/scale, float(obj["top"])/scale
        w = float(obj["width"])*float(obj.get("scaleX", 1))/scale
        h = float(obj["height"])*float(obj.get("scaleY", 1))/scale
        if min(x,y) < 0 or min(w,h) <= 0 or x+w > image_width+1 or y+h > image_height+1:
            raise ValueError("Box outside original image")
        boxes.append([round(x),round(y),round(min(w,image_width-x)),round(min(h,image_height-y))])
    return boxes