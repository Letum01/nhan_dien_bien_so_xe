"""Read executed train-only artifacts; preserve validation as historical evidence."""
import argparse
import json
from pathlib import Path


def f1(record):
    s=record['segmentation']
    return 2*s['tp']/max(1,2*s['tp']+s['fp']+s['fn'])


def main(folder):
    folder=Path(folder);data=json.loads((folder/'results.json').read_text(encoding='utf-8'))
    selected=data['selected'];metrics=data['variants']
    lines=['# Targeted segmentation ablation — TRAIN DEVELOPMENT ONLY',
           '\n## Protocol / safety',
           '18trainplates147glyphs,frozenYOLOcropswith15%padding frompreviousbenchmark. '
           'OnlyTRAINimagesopenedforcandidateablation;7validationnotrerunorusedforparameterchoice. '
           'IoU>=.5maximum-cardinalityone-to-oneafterinference;noGTsplit/merge/textrepair. '
           'FrozenCNNMixloadedforclassification,nottrained. Allproductioncrop/segmentation/weights/threshold unchanged.',
           '\n## Five predefined variants',
           '- frozen_border: existing15%+border_components experimentalreference.',
           '- protected_border:retainordinaryedge-touchingcomponents;removeonlyspanning/thinframe. MoreFPpossible.',
           '- geometry:removecomponentsinconsistentwithmedianheightandnarrowaspect. Mayeraselegitimate1/shortglyph.',
           '- projection_split:widecomponentprojectionvalley,notexpectedtextcount. Noeffectobservedonthese18crops.',
           '- row_sort:largestcenter-Ygapscaledbymedianglyphheight. Noeffectobservedonthese18crops.',
           '\n|Variant|TP|FP|FN|Precision|Recall|F1|Count equal|Full order|Exact text|Matched CNN|',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,m in metrics.items():
        lines.append(f"|{name}|{m['tp']}|{m['fp']}|{m['fn']}|{m['precision']:.4f}|{m['recall']:.4f}|{m['f1']:.4f}|{m['same_glyph_count_plates']}/18|{m['full_order_plates']}/18|{m['exact_plate_count']}/18|{m['matched_classifier_correct']}/{m['matched_classifier_samples']}|")
    lines += [f'\nTrainF1winner={selected},butNOTPROMOTED.FP13→8,FN18→20,exact10→8. '
              'A single segmentationaggregate improvement does not establish better ALPR or safety for boundary glyphs.',
              '\n## Per TRAIN plate: selected versus reference',
              '|Image|Before TP/FP/FN|After TP/FP/FN|Seg F1 change|Before text|After text|',
              '|---|---|---|---|---|---|']
    changes=[]
    for old in [r for r in data['records'] if r['config']=='frozen_border']:
        new=next(r for r in data['records'] if r['plate_id']==old['plate_id'] and r['config']==selected)
        state='improved' if f1(new)>f1(old) else 'regressed' if f1(new)<f1(old) else 'same'
        a=old['segmentation'];b=new['segmentation']
        changes.append({'image':old['image'],'state':state,'before':{k:a[k] for k in ('tp','fp','fn')},
                        'after':{k:b[k] for k in ('tp','fp','fn')},'before_exact':old['exact_text'],'after_exact':new['exact_text']})
        lines.append(f"|{Path(old['image']).name}|{a['tp']}/{a['fp']}/{a['fn']}|{b['tp']}/{b['fp']}/{b['fn']}|{state}|{old['prediction']}|{new['prediction']}|")
    lines += ['\nGeometryregression:05016.jpg8TP0FP0FN→7TP0FP1FN;0000_00532_b.jpg8TP1FP1FN→7TP0FP2FN. '
              'These showgeometryfiltercanremovevalidglyphs. ProtectedborderkeepssameTP129butaddsFP13→19andexact10→8. '
              'No validation rescue/tuning performed. Keepallvariants experimental.',
              '\n## Error analysis per plate (previous frozen benchmark, not newVALexperiment)',
              'The following uses storedp15_border_componentsrecords fromcrop_paddingbenchmark. '
              'Cropcoverage=allGTglyphfullycontained;unmatchedpredboxeslistedglobalxywh;GTmissingindices0based; '
              'matchclassificationerrorsonlyreportedaftervalidIoUmatch,notbyindex. '
              'Split/mergeoverlapcandidatesinCSVarehypotheses,notautomaticcausalannotations.',
              '\n|Image|Split|Coverage|Extra prediction boxes|Missing GT index|Complete order|Matched classification errors|',
              '|---|---|---|---|---|---|---|']
    for error in data['previous_frozen_error_analysis']:
        lines.append(f"|{Path(error['image']).name}|{error['split']}|{error['coverage']}|{error['unmatched_noise_boxes']}|{error['missing_gt_indices']}|{error['complete_order_correct']}|{error['classification_errors_on_matches']}|")
    lines += ['\nFailurecategoriesarenotexclusive:allpad15cropscontainGTglyphs,butconnectedink/frames/illuminationcan '
              'stillmergesplit/removecomponents. Someexacttextscoincidewithunmatchedgeometry(e.g.0416),which '
              'doesnotprovecomponentboxesarevalid. EqualcountnotproofandfullorderrequiresfullIoUmatching.',
              '\n## Evidence and additional review queue',
              'Eachtrainplatefolder:fiveoverlayPNGandbinaryPNG(greenGT/redcandidate). '
              'results.jsonandtrain_comparison.csvholdmatchedindices/globalboxes,predictions,split/mergecandidates. '
              'frozen_per_plate_errors.csvisall25platehistoricalerrorinventory. No GT altered.',
              '28remainingTRAINDetectionreviewedplateswithnoGlyphreviewedrankedinreports/ocr_hard_queue_20261125T091023673153Z/. '
              'Smallarea,laplacianblurproxy,borderink,edge-rectangleangleareUNVERIFIEDprioritymetadata. '
              'Classcoverageusesonlyexistingconfirmedhumantext;unknowntextstaysunknown. Allgroup/splitinheritedtrain.',
              '\n## Next steps',
              'Humanreviewthetopqueueespeciallysmall/blurred/tiltedplatesandframesnearink. Labelseparateglyphs '
              'withoutcopyingmodelpredictions;fixsplit/mergedproposalsandretainoriginalsourcegroups. '
              'Foralgorithmdevelopment,annotateframe/noisecomponentsandtouchingglyphcases,thenmeasurefalse '
              'removalonnewsource-groupholdouts. Current18trainexamplesalreadyreusedinmanyexperiments; '
              'overfitriskhigh. No furthercheckof7VALforselectionandnoOCRtestavailable.']
    (folder/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    (folder/'train_improvements_regressions.json').write_text(json.dumps(changes,indent=2),encoding='utf-8')
    print(folder/'REPORT.md')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);args=parser.parse_args();main(args.folder)