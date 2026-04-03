import 'package:flutter/material.dart';
import 'dart:ui';
import 'dart:math' as math;
import 'package:google_fonts/google_fonts.dart';
import '../services/feed_state.dart';
import 'dart:io';

const _kSurface = Color(0xFF141416);
const _kCard = Color(0xFF1C1C1F);
const _kBorder = Color(0xFF27272A);
const _kAccent = Color(0xFFC9A84C);
const _kAccent2 = Color(0xFFF0D78C);
const _kMuted = Color(0xFF71717A);

class ModelScreen extends StatefulWidget {
  const ModelScreen({super.key});
  @override
  State<ModelScreen> createState() => _ModelScreenState();
}

class _ModelScreenState extends State<ModelScreen> with SingleTickerProviderStateMixin {
  late AnimationController _animCtrl;

  @override
  void initState() {
    super.initState();
    _animCtrl = AnimationController(vsync: this, duration: const Duration(milliseconds: 1500))..forward();
  }

  @override
  void dispose() { _animCtrl.dispose(); super.dispose(); }

  @override
  Widget build(BuildContext context) {
    final feedState = FeedStateProvider.of(context);
    return SingleChildScrollView(
      physics: const BouncingScrollPhysics(),
      padding: const EdgeInsets.fromLTRB(16, 16, 16, 120),
      child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
        _buildAurocGauge(),
        const SizedBox(height: 24),
        _buildMetricCards(),
        const SizedBox(height: 24),
        _buildBenchmarkChart(),
        const SizedBox(height: 24),
        _buildConfusionMatrix(),
        const SizedBox(height: 24),
        _buildLossCurve(),
        const SizedBox(height: 24),
        _buildScanHistory(feedState),
      ]),
    );
  }

  Widget _buildAurocGauge() {
    return Container(
      padding: const EdgeInsets.all(28),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(24),
        gradient: LinearGradient(begin: Alignment.topLeft, end: Alignment.bottomRight, colors: [_kAccent.withOpacity(0.06), _kSurface, _kAccent.withOpacity(0.03)]),
        border: Border.all(color: _kAccent.withOpacity(0.12)),
      ),
      child: Column(children: [
        Text('MODEL PERFORMANCE', style: GoogleFonts.outfit(color: _kAccent, fontSize: 12, fontWeight: FontWeight.w700, letterSpacing: 1.5)),
        const SizedBox(height: 20),
        AnimatedBuilder(animation: _animCtrl, builder: (_, __) {
          return SizedBox(width: 160, height: 160, child: CustomPaint(painter: _GaugePainter(_animCtrl.value * 0.82)));
        }),
        const SizedBox(height: 16),
        ShaderMask(
          shaderCallback: (b) => const LinearGradient(colors: [_kAccent, _kAccent2]).createShader(b),
          child: Text('82% AUROC', style: GoogleFonts.outfit(fontSize: 28, fontWeight: FontWeight.w800, color: Colors.white)),
        ),
        const SizedBox(height: 6),
        Text('Area Under ROC Curve', style: GoogleFonts.inter(color: _kMuted, fontSize: 12)),
        const SizedBox(height: 4),
        Text('Facebook Hateful Memes Dataset', style: GoogleFonts.inter(color: _kMuted.withOpacity(0.6), fontSize: 11)),
      ]),
    );
  }

  Widget _buildMetricCards() {
    final metrics = [
      ('Precision', '0.80', const Color(0xFF60a5fa)),
      ('Recall', '0.80', const Color(0xFF34d399)),
      ('F1-Score', '0.80', const Color(0xFF818cf8)),
      ('Accuracy', '0.80', _kAccent),
    ];
    return Row(children: metrics.map((m) => Expanded(child: Container(
      margin: const EdgeInsets.symmetric(horizontal: 4),
      padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 8),
      decoration: BoxDecoration(
        color: m.$3.withOpacity(0.05), borderRadius: BorderRadius.circular(16),
        border: Border.all(color: m.$3.withOpacity(0.15)),
      ),
      child: Column(children: [
        Text(m.$2, style: GoogleFonts.outfit(color: m.$3, fontSize: 20, fontWeight: FontWeight.w800)),
        const SizedBox(height: 4),
        Text(m.$1, style: GoogleFonts.inter(color: _kMuted, fontSize: 10, fontWeight: FontWeight.w500)),
      ]),
    ))).toList());
  }

  Widget _buildBenchmarkChart() {
    final models = [
      ('Sentinel-X\n(Ours)', 0.82, true),
      ('ViLBERT', 0.73, false),
      ('Visual BERT', 0.71, false),
      ('Text Only\n(BiLSTM)', 0.65, false),
      ('Image Only\n(CNN)', 0.52, false),
    ];
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      _sec('BENCHMARK COMPARISON', Icons.leaderboard_outlined),
      const SizedBox(height: 14),
      Container(
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(color: _kSurface, borderRadius: BorderRadius.circular(18), border: Border.all(color: _kBorder.withOpacity(0.6))),
        child: Column(children: models.map((m) {
          final (name, score, isOurs) = m;
          return Padding(padding: const EdgeInsets.symmetric(vertical: 6), child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
              SizedBox(width: 80, child: Text(name, style: GoogleFonts.inter(color: isOurs ? _kAccent : Colors.white.withOpacity(0.6), fontSize: 10, fontWeight: isOurs ? FontWeight.w700 : FontWeight.w400, height: 1.3))),
              const SizedBox(width: 12),
              Expanded(child: AnimatedBuilder(animation: _animCtrl, builder: (_, __) {
                return ClipRRect(borderRadius: BorderRadius.circular(4), child: LinearProgressIndicator(
                  value: score * _animCtrl.value, minHeight: isOurs ? 12 : 8,
                  backgroundColor: _kBorder.withOpacity(0.4),
                  valueColor: AlwaysStoppedAnimation<Color>(isOurs ? _kAccent : _kMuted.withOpacity(0.5)),
                ));
              })),
              const SizedBox(width: 10),
              Text('${(score * 100).toInt()}%', style: GoogleFonts.firaCode(color: isOurs ? _kAccent : _kMuted, fontSize: 11, fontWeight: FontWeight.w600)),
            ]),
          ]));
        }).toList()),
      ),
    ]);
  }

  Widget _buildConfusionMatrix() {
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      _sec('CLASSIFICATION REPORT', Icons.assessment_outlined),
      const SizedBox(height: 14),
      Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(color: _kSurface, borderRadius: BorderRadius.circular(18), border: Border.all(color: _kBorder.withOpacity(0.6))),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          _reportHeader(),
          const SizedBox(height: 8),
          _reportRow('Safe', '0.77', '0.85', '0.81', '250', const Color(0xFF4ade80)),
          const SizedBox(height: 6),
          _reportRow('Hateful', '0.84', '0.75', '0.79', '250', const Color(0xFFf87171)),
          Padding(padding: const EdgeInsets.symmetric(vertical: 8), child: Divider(color: _kBorder.withOpacity(0.4), height: 1)),
          _reportRow('Weighted Avg', '0.80', '0.80', '0.80', '500', _kAccent, bold: true),
        ]),
      ),
      const SizedBox(height: 16),
      _sec('CONFUSION MATRIX', Icons.grid_on_outlined),
      const SizedBox(height: 14),
      Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(color: _kSurface, borderRadius: BorderRadius.circular(18), border: Border.all(color: _kBorder.withOpacity(0.6))),
        child: Column(children: [
          Row(children: [
            const SizedBox(width: 60),
            Expanded(child: Text('Predicted\nSafe', textAlign: TextAlign.center, style: GoogleFonts.inter(color: _kMuted, fontSize: 10, height: 1.3))),
            Expanded(child: Text('Predicted\nHateful', textAlign: TextAlign.center, style: GoogleFonts.inter(color: _kMuted, fontSize: 10, height: 1.3))),
          ]),
          const SizedBox(height: 8),
          Row(children: [
            SizedBox(width: 60, child: Text('Actually\nSafe', style: GoogleFonts.inter(color: _kMuted, fontSize: 10, height: 1.3))),
            _cmCell('213', const Color(0xFF22c55e), 'TN'),
            const SizedBox(width: 6),
            _cmCell('37', const Color(0xFFef4444).withOpacity(0.5), 'FP'),
          ]),
          const SizedBox(height: 6),
          Row(children: [
            SizedBox(width: 60, child: Text('Actually\nHateful', style: GoogleFonts.inter(color: _kMuted, fontSize: 10, height: 1.3))),
            _cmCell('63', const Color(0xFFef4444).withOpacity(0.5), 'FN'),
            const SizedBox(width: 6),
            _cmCell('187', const Color(0xFF22c55e), 'TP'),
          ]),
        ]),
      ),
    ]);
  }

  Widget _reportHeader() {
    final style = GoogleFonts.firaCode(color: _kMuted, fontSize: 10, fontWeight: FontWeight.w600);
    return Row(children: [
      SizedBox(width: 80, child: Text('Class', style: style)),
      Expanded(child: Text('Prec', textAlign: TextAlign.center, style: style)),
      Expanded(child: Text('Recall', textAlign: TextAlign.center, style: style)),
      Expanded(child: Text('F1', textAlign: TextAlign.center, style: style)),
      SizedBox(width: 50, child: Text('Supp', textAlign: TextAlign.right, style: style)),
    ]);
  }

  Widget _reportRow(String cls, String prec, String rec, String f1, String supp, Color c, {bool bold = false}) {
    final style = GoogleFonts.firaCode(color: c, fontSize: 12, fontWeight: bold ? FontWeight.w700 : FontWeight.w500);
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 4),
      decoration: BoxDecoration(color: c.withOpacity(0.04), borderRadius: BorderRadius.circular(8)),
      child: Row(children: [
        SizedBox(width: 76, child: Text(cls, style: GoogleFonts.inter(color: c, fontSize: 11, fontWeight: bold ? FontWeight.w700 : FontWeight.w600))),
        Expanded(child: Text(prec, textAlign: TextAlign.center, style: style)),
        Expanded(child: Text(rec, textAlign: TextAlign.center, style: style)),
        Expanded(child: Text(f1, textAlign: TextAlign.center, style: style)),
        SizedBox(width: 50, child: Text(supp, textAlign: TextAlign.right, style: GoogleFonts.firaCode(color: _kMuted, fontSize: 11))),
      ]),
    );
  }

  Widget _cmCell(String val, Color c, String label) => Expanded(child: Container(
    padding: const EdgeInsets.symmetric(vertical: 16),
    decoration: BoxDecoration(color: c.withOpacity(0.08), borderRadius: BorderRadius.circular(12), border: Border.all(color: c.withOpacity(0.2))),
    child: Column(children: [
      Text(val, style: GoogleFonts.outfit(color: c, fontSize: 22, fontWeight: FontWeight.w800)),
      const SizedBox(height: 2),
      Text(label, style: GoogleFonts.firaCode(color: c.withOpacity(0.6), fontSize: 10)),
    ]),
  ));

  Widget _buildLossCurve() {
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      _sec('TRAINING LOSS CURVE', Icons.show_chart),
      const SizedBox(height: 14),
      Container(
        height: 180,
        decoration: BoxDecoration(color: _kSurface, borderRadius: BorderRadius.circular(18), border: Border.all(color: _kBorder.withOpacity(0.6))),
        child: AnimatedBuilder(animation: _animCtrl, builder: (_, __) {
          return ClipRRect(borderRadius: BorderRadius.circular(18), child: CustomPaint(painter: _LossCurvePainter(_animCtrl.value), size: Size.infinite));
        }),
      ),
    ]);
  }

  Widget _buildScanHistory(FeedState feedState) {
    final history = feedState.scanHistory;
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      _sec('RECENT SCANS', Icons.history),
      const SizedBox(height: 14),
      if (history.isEmpty)
        Container(
          padding: const EdgeInsets.all(32),
          decoration: BoxDecoration(color: _kSurface, borderRadius: BorderRadius.circular(18), border: Border.all(color: _kBorder.withOpacity(0.6))),
          child: Center(child: Column(children: [
            Icon(Icons.document_scanner_outlined, color: _kMuted.withOpacity(0.4), size: 40),
            const SizedBox(height: 12),
            Text('No scans yet', style: GoogleFonts.inter(color: _kMuted, fontSize: 14)),
            const SizedBox(height: 4),
            Text('Upload a meme to see scan results here', style: GoogleFonts.inter(color: _kMuted.withOpacity(0.5), fontSize: 12)),
          ])),
        )
      else
        ...history.map((h) {
          final scorePercent = (h.toxicityScore * 100).toStringAsFixed(1);
          final sc = h.isSafe ? const Color(0xFF22c55e) : const Color(0xFFef4444);
          final timeAgo = _timeAgo(h.timestamp);
          return Container(
            margin: const EdgeInsets.only(bottom: 8),
            padding: const EdgeInsets.all(14),
            decoration: BoxDecoration(color: _kSurface, borderRadius: BorderRadius.circular(14), border: Border.all(color: _kBorder.withOpacity(0.6))),
            child: Row(children: [
              ClipRRect(borderRadius: BorderRadius.circular(10), child: File(h.imagePath).existsSync()
                  ? Image.file(File(h.imagePath), width: 48, height: 48, fit: BoxFit.cover)
                  : Container(width: 48, height: 48, color: _kCard, child: const Icon(Icons.broken_image, color: _kMuted, size: 20))),
              const SizedBox(width: 14),
              Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Text(h.isSafe ? 'Content Approved' : 'Content Flagged', style: GoogleFonts.inter(color: sc, fontSize: 13, fontWeight: FontWeight.w600)),
                Text(timeAgo, style: GoogleFonts.inter(color: _kMuted, fontSize: 11)),
              ])),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
                decoration: BoxDecoration(color: sc.withOpacity(0.08), borderRadius: BorderRadius.circular(8)),
                child: Text('$scorePercent%', style: GoogleFonts.firaCode(color: sc, fontSize: 12, fontWeight: FontWeight.w600)),
              ),
            ]),
          );
        }),
    ]);
  }

  String _timeAgo(DateTime dt) {
    final diff = DateTime.now().difference(dt);
    if (diff.inMinutes < 1) return 'Just now';
    if (diff.inMinutes < 60) return '${diff.inMinutes}m ago';
    if (diff.inHours < 24) return '${diff.inHours}h ago';
    return '${diff.inDays}d ago';
  }

  Widget _sec(String t, IconData ic) => Row(children: [Icon(ic, color: _kAccent, size: 16), const SizedBox(width: 8), Text(t, style: GoogleFonts.outfit(color: _kAccent, fontSize: 12, fontWeight: FontWeight.w700, letterSpacing: 1.5))]);
}

class _GaugePainter extends CustomPainter {
  final double progress;
  _GaugePainter(this.progress);

  @override
  void paint(Canvas canvas, Size size) {
    final c = Offset(size.width / 2, size.height / 2);
    final r = size.width / 2 - 12;
    canvas.drawArc(Rect.fromCircle(center: c, radius: r), 0.8, math.pi * 1.4, false,
        Paint()..color = const Color(0xFF27272A)..strokeWidth = 10..style = PaintingStyle.stroke..strokeCap = StrokeCap.round);
    if (progress > 0) {
      final sweep = math.pi * 1.4 * progress;
      final grad = SweepGradient(startAngle: 0.8, endAngle: 0.8 + sweep, colors: const [_kAccent, _kAccent2])
          .createShader(Rect.fromCircle(center: c, radius: r));
      canvas.drawArc(Rect.fromCircle(center: c, radius: r), 0.8, sweep, false,
          Paint()..shader = grad..strokeWidth = 10..style = PaintingStyle.stroke..strokeCap = StrokeCap.round);
    }
  }

  @override
  bool shouldRepaint(_GaugePainter o) => o.progress != progress;
}

class _LossCurvePainter extends CustomPainter {
  final double progress;
  _LossCurvePainter(this.progress);

  @override
  void paint(Canvas canvas, Size size) {
    final w = size.width; final h = size.height;
    final pad = 30.0;
    canvas.drawLine(Offset(pad, h - pad), Offset(w - 10, h - pad), Paint()..color = const Color(0xFF27272A)..strokeWidth = 0.5);
    canvas.drawLine(Offset(pad, 10), Offset(pad, h - pad), Paint()..color = const Color(0xFF27272A)..strokeWidth = 0.5);

    final trainPts = <Offset>[];
    final valPts = <Offset>[];
    const epochs = 30;
    for (int i = 0; i <= epochs; i++) {
      final x = pad + (w - pad - 10) * (i / epochs);
      final t = i / epochs;
      final trainY = h - pad - (h - pad - 20) * (0.18 + 0.72 * math.exp(-3.5 * t));
      final valY = h - pad - (h - pad - 20) * (0.22 + 0.68 * math.exp(-2.8 * t) + 0.03 * math.sin(t * 8));
      trainPts.add(Offset(x, trainY));
      valPts.add(Offset(x, valY));
    }

    final visibleTrain = (trainPts.length * progress).round().clamp(2, trainPts.length);
    final visibleVal = (valPts.length * progress).round().clamp(2, valPts.length);

    _drawCurve(canvas, trainPts.sublist(0, visibleTrain), _kAccent);
    _drawCurve(canvas, valPts.sublist(0, visibleVal), const Color(0xFF818cf8));

    final tp1 = TextPainter(text: TextSpan(text: 'Epochs', style: GoogleFonts.inter(color: _kMuted, fontSize: 9)), textDirection: TextDirection.ltr)..layout();
    tp1.paint(canvas, Offset(w / 2 - tp1.width / 2, h - 12));
    final tp2 = TextPainter(text: TextSpan(text: '— Train  ', style: GoogleFonts.inter(color: _kAccent, fontSize: 9)), textDirection: TextDirection.ltr)..layout();
    tp2.paint(canvas, Offset(w - 120, 8));
    final tp3 = TextPainter(text: TextSpan(text: '— Val', style: GoogleFonts.inter(color: const Color(0xFF818cf8), fontSize: 9)), textDirection: TextDirection.ltr)..layout();
    tp3.paint(canvas, Offset(w - 55, 8));
  }

  void _drawCurve(Canvas canvas, List<Offset> pts, Color color) {
    if (pts.length < 2) return;
    final path = Path()..moveTo(pts[0].dx, pts[0].dy);
    for (int i = 1; i < pts.length; i++) {
      final cp1 = Offset((pts[i - 1].dx + pts[i].dx) / 2, pts[i - 1].dy);
      final cp2 = Offset((pts[i - 1].dx + pts[i].dx) / 2, pts[i].dy);
      path.cubicTo(cp1.dx, cp1.dy, cp2.dx, cp2.dy, pts[i].dx, pts[i].dy);
    }
    canvas.drawPath(path, Paint()..color = color..strokeWidth = 2..style = PaintingStyle.stroke..strokeCap = StrokeCap.round);
  }

  @override
  bool shouldRepaint(_LossCurvePainter o) => o.progress != progress;
}
