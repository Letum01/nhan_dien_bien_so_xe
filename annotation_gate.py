"""Frozen annotation membership gate retained by Character training."""

def check_gate(protocol,reviewed):
    rows={k:r for k,r in reviewed.items() if r.get('review_status')=='reviewed'}
    missing=[r['image_path'] for r in protocol['plates'] if r['image_id'] not in rows]
    if missing:raise ValueError('BLOCKED: hard13 still missing reviewed glyphs: '+str(missing))
    for expected in protocol['plates']:
        current=rows[expected['image_id']]
        for key in ('source_image_id','plate_bbox','source_group','split','detection_revision'):
            if current[key]!=expected[key]:raise ValueError('Hard13parentchanged:'+expected['image_path'])
    return [rows[r['image_id']] for r in protocol['plates']]
