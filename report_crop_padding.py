"""Report locked train-selected padding with validation regressions preserved."""
import argparse
import json
from pathlib import Path


def main(folder):
    folder=Path(folder);d=json.loads((folder/'results.json').read_text(encoding='utf-8'))
    selected=d['selected_train_only']
    lines=['# YOLO crop padding experiment',
           '\nWorkspace:C:/Users/Admin/Documents/nhan_dien_bien_so_xe (nan_dien directory absent). '
           'Current adapter rounds predicted xyxy to integerclippedxywh and crops WITHOUTpadding. '
           'This experiment retains originalbox/score and adds0/5/10/15%oforiginalw/h PER SIDE, '
           'floorleft/top,ceilright/bottom,clippedimagebounds. Inference selects topdetectorscore,neverGT.',
           '\nSelection frozen on18train147glyph segmentationF1only before7val57glyph run; '
           f'selected **{selected}**. SameCNNMixcheckpoint,threshold.4,mapping/preprocessing. '
           'No training,GToutputrepair,productionsegmentationorcropchanges. Developmentset repeatedfromearlierexperiments,notfinaltest.',
           '\n|Split|Config|Full glyph coverage|Full crop plates|TP|FP|FN|F1|Count equal|Full order|Matched CNN|Exact text|',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for split,configs in d['summary'].items():
        for name,r in configs.items():
            lines.append(f"|{split}|{name}|{r['fully_covered_glyphs']}|{r['crop_full_coverage_plates']}/{r['plates']}|{r['tp']}|{r['fp']}|{r['fn']}|{r['f1']:.4f}|{r['same_glyph_count_plates']}/{r['plates']}|{r['full_order_plates']}/{r['plates']}|{r['matched_classifier_correct']}/{r['matched_classifier_samples']}|{r['exact_plate_count']}/{r['plates']}|")
    lines+=['\n## Per validation image: baseline vs frozen selected',
            '|Image|Config|Coverage|TP/FP/FN|Text|Reviewed text|Exact|','|---|---|---:|---|---|---|---|']
    changes=[]
    for base in [r for r in d['records'] if r['split']=='val' and r['config']=='p00_baseline']:
        new=next(r for r in d['records'] if r['plate_id']==base['plate_id'] and r['config']==selected)
        for r in (base,new):
            m=r['segmentation']
            lines.append(f"|{Path(r['image']).name}|{r['config']}|{r['fully_covered_glyphs']}/{r['gt_glyph_count']}|{m['tp']}/{m['fp']}/{m['fn']}|{r['prediction']}|{r['reviewed_text']}|{r['exact_text']}|")
        def f1(r):
            s=r['segmentation'];return 2*s['tp']/max(1,2*s['tp']+s['fp']+s['fn'])
        changes.append({'image':base['image'],'segmentation_change':'improved' if f1(new)>f1(base) else 'regressed' if f1(new)<f1(base) else 'same',
                        'baseline_text':base['prediction'],'candidate_text':new['prediction'],
                        'baseline_exact':base['exact_text'],'candidate_exact':new['exact_text']})
    (folder/'validation_changes.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2),encoding='utf-8')
    lines+=['\n## Findings / limits',
            'Padding15 restorescoverageall57glyphsin7valplates vs54glyphs/6fullplates at0%. '
            'Selectedsegmentation53TP/6FP/4FN improvesF1,butfullorderonly3/7 andexacttext1/7versusbaseline1/7. '
            'Thus improvement in coverage/segmentation doesNOTjustify OCRproductionpromotion. '
            'MatchedCNN49/53mustbereportedwith4FNand6FP;unmatchednoise/missingglyphsnotclassifiercorrect.',
            'Padding enlargesbackground/framesandchangesCLAHE/Otsustatistics;baselinewithpaddingcanloseconnectedglyphs. '
            'Border-connectedfilter caneraseedgeglyphs,particularlywheninkconnectstoframe. '
            'Not all paddinghelps;allnegativecases retained. No choices madebyvalexacttext.',
            'Previoussegmentationbenchmarkreported2/7for0%bordercomponents;thisfreshmatchedrunreproduces1/7baselineand2/7zero-paddingbordercomponents. '
            'Useonlywithin-runmetricsforthepaddingcomparison;doNOTclaim1→2or2→1withoutmatchingconfiguration. '
            'Earlier2/7referredto0%bordercomponents,notproduction0%baseline. '
            'Detectionmetadata/checkpointSHAandallcropglobalcoordinatesareinresults.json/protocol.json.',
            '\nArtifacts:train/valplatefolders containcrop,binary,contourtraceJSON,fullimageoverlay '
            '(greenGTglyph,redpredglyph,yelloworiginalYOLO,blueexpandedcrop). '
            'GTglobalcoordinatesonlyusedafteroutputforIoUcoverageandclassscore,nevertocorrectinference. '
            'No testpilotused. Timingsincludedetectorstartup/cacheandper-cropCNNpredict overhead,notrealtimeFPS.',
            '\nDecision:keepselectedcandidateexperimental,keepallproductionsettingsunchanged. '
            'Needindependentnewreviewedcrop/glyphsourcesandrow/framehandlingbenchmarkbeforepromotion.']
    (folder/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(folder/'REPORT.md')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);args=parser.parse_args();main(args.folder)