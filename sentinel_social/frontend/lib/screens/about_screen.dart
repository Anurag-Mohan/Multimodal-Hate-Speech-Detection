import 'package:flutter/material.dart';
import 'dart:ui';
import 'package:google_fonts/google_fonts.dart';

const _kBg = Color(0xFF09090B);
const _kSurface = Color(0xFF141416);
const _kBorder = Color(0xFF27272A);
const _kAccent = Color(0xFFC9A84C);
const _kAccent2 = Color(0xFFF0D78C);
const _kMuted = Color(0xFF71717A);

class AboutScreen extends StatefulWidget {
  const AboutScreen({super.key});
  @override
  State<AboutScreen> createState() => _AboutScreenState();
}

class _AboutScreenState extends State<AboutScreen> with SingleTickerProviderStateMixin {
  late AnimationController _pulseCtrl;

  @override
  void initState() {
    super.initState();
    _pulseCtrl = AnimationController(vsync: this, duration: const Duration(milliseconds: 2000))..repeat(reverse: true);
  }

  @override
  void dispose() {
    _pulseCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      physics: const BouncingScrollPhysics(),
      padding: const EdgeInsets.fromLTRB(16, 16, 16, 120),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          _buildHero(),
          const SizedBox(height: 28),
          _buildProblem(),
          const SizedBox(height: 24),
          _buildConfounder(),
          const SizedBox(height: 24),
          _buildSolution(),
          const SizedBox(height: 24),
          _buildPipeline(),
          const SizedBox(height: 24),
          _buildTraining(),
          const SizedBox(height: 24),
          _buildWhyMatters(),
          const SizedBox(height: 24),
          _buildTech(),
        ],
      ),
    );
  }

  Widget _buildHero() {
    return AnimatedBuilder(
      animation: _pulseCtrl,
      builder: (_, __) {
        return Container(
          padding: const EdgeInsets.symmetric(vertical: 40, horizontal: 20),
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(24),
            gradient: LinearGradient(begin: Alignment.topLeft, end: Alignment.bottomRight, colors: [_kAccent.withOpacity(0.06), _kSurface, _kAccent.withOpacity(0.03)]),
            border: Border.all(color: _kAccent.withOpacity(0.12)),
            boxShadow: [BoxShadow(color: _kAccent.withOpacity(0.08 + _pulseCtrl.value * 0.12), blurRadius: 60, spreadRadius: -10)],
          ),
          child: Column(children: [
            Container(
              width: 80, height: 80,
              decoration: BoxDecoration(shape: BoxShape.circle, gradient: LinearGradient(colors: [_kAccent.withOpacity(0.15), _kAccent.withOpacity(0.05)]), border: Border.all(color: _kAccent.withOpacity(0.3))),
              child: const Icon(Icons.shield_rounded, color: _kAccent, size: 38),
            ),
            const SizedBox(height: 20),
            ShaderMask(
              shaderCallback: (b) => const LinearGradient(colors: [_kAccent, _kAccent2]).createShader(b),
              child: Text('Why Sentinel-X?', style: GoogleFonts.outfit(fontSize: 28, fontWeight: FontWeight.w800, color: Colors.white)),
            ),
            const SizedBox(height: 12),
            Text('Defending digital spaces from multimodal hate speech\nthrough intelligent AI-powered content moderation.', textAlign: TextAlign.center, style: GoogleFonts.inter(color: Colors.white.withOpacity(0.5), fontSize: 14, height: 1.6)),
          ]),
        );
      },
    );
  }

  Widget _buildProblem() {
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      _sec('THE PROBLEM', Icons.warning_amber_rounded),
      const SizedBox(height: 14),
      _glass(Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Text('Weaponization of Visual Culture', style: GoogleFonts.outfit(color: Colors.white, fontSize: 17, fontWeight: FontWeight.w700)),
        const SizedBox(height: 10),
        Text('Digital communication has shifted to a visually dominant culture driven by memes. While this enriches expression, it creates sophisticated vectors for disseminating hate speech and extremist propaganda that bypass standard text-based content moderation filters.',
            style: GoogleFonts.inter(color: Colors.white.withOpacity(0.55), fontSize: 13, height: 1.65)),
        const SizedBox(height: 16),
        Row(children: [_stat('82%', 'hate memes\nbypass filters'), const SizedBox(width: 10), _stat('3.2B+', 'memes shared\ndaily'), const SizedBox(width: 10), _stat('< 5%', 'detected by\ntraditional AI')]),
      ])),
    ]);
  }

  Widget _stat(String v, String l) => Expanded(child: Container(
    padding: const EdgeInsets.all(12),
    decoration: BoxDecoration(color: _kAccent.withOpacity(0.06), borderRadius: BorderRadius.circular(12), border: Border.all(color: _kAccent.withOpacity(0.12))),
    child: Column(children: [
      ShaderMask(shaderCallback: (b) => const LinearGradient(colors: [_kAccent, _kAccent2]).createShader(b), child: Text(v, style: GoogleFonts.outfit(color: Colors.white, fontSize: 18, fontWeight: FontWeight.w800))),
      const SizedBox(height: 4),
      Text(l, textAlign: TextAlign.center, style: GoogleFonts.inter(color: _kMuted, fontSize: 9.5, height: 1.3)),
    ]),
  ));

  Widget _buildConfounder() {
    return _glass(Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Row(children: [
        Container(padding: const EdgeInsets.all(8), decoration: BoxDecoration(color: const Color(0xFFef4444).withOpacity(0.08), borderRadius: BorderRadius.circular(10)), child: const Icon(Icons.psychology_outlined, color: Color(0xFFf87171), size: 20)),
        const SizedBox(width: 12),
        Expanded(child: Text('The Benign Confounder Problem', style: GoogleFonts.outfit(color: const Color(0xFFfca5a5), fontSize: 15, fontWeight: FontWeight.w700))),
      ]),
      const SizedBox(height: 14),
      Text('The image and caption are innocuous in isolation. Traditional unimodal models are blind to the "semantic friction" that creates toxicity exclusively at the intersection.', style: GoogleFonts.inter(color: Colors.white.withOpacity(0.5), fontSize: 12.5, height: 1.6)),
      const SizedBox(height: 14),
      Container(
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(color: _kBg, borderRadius: BorderRadius.circular(12), border: Border.all(color: _kBorder.withOpacity(0.6))),
        child: Row(children: [
          _confItem(Icons.image_outlined, 'Image', 'Safe ✓', const Color(0xFF4ade80)),
          Text('+', style: GoogleFonts.outfit(color: _kMuted, fontSize: 20)),
          _confItem(Icons.text_fields, 'Text', 'Safe ✓', const Color(0xFF4ade80)),
          Text('=', style: GoogleFonts.outfit(color: _kMuted, fontSize: 20)),
          _confItem(Icons.dangerous_outlined, 'Meme', 'Toxic ✗', const Color(0xFFf87171)),
        ]),
      ),
    ]), bc: const Color(0xFFef4444).withOpacity(0.15));
  }

  Widget _confItem(IconData ic, String t, String s, Color c) => Expanded(child: Column(children: [
    Icon(ic, color: c, size: 24), const SizedBox(height: 6),
    Text(t, style: GoogleFonts.firaCode(color: c, fontSize: 10)),
    Text(s, style: GoogleFonts.firaCode(color: c, fontSize: 9)),
  ]));

  Widget _buildSolution() {
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      _sec('OUR SOLUTION', Icons.auto_awesome),
      const SizedBox(height: 14),
      _glass(Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        ShaderMask(shaderCallback: (b) => const LinearGradient(colors: [_kAccent, _kAccent2]).createShader(b),
          child: Text('Attentive Hybrid Multimodal Architecture', style: GoogleFonts.outfit(color: Colors.white, fontSize: 16, fontWeight: FontWeight.w700))),
        const SizedBox(height: 12),
        Text('A specialized dual-stream pipeline extracts and fuses deep semantic meaning across modalities without massive computational overhead.', style: GoogleFonts.inter(color: Colors.white.withOpacity(0.5), fontSize: 13, height: 1.6)),
        const SizedBox(height: 16),
        _stream('Visual Stream', Icons.remove_red_eye_outlined, ['Custom CNN + 2D Batch Normalization', 'CLIP ViT-B/32 zero-shot embeddings'], const Color(0xFF818cf8)),
        const SizedBox(height: 10),
        _stream('Text Stream', Icons.text_snippet_outlined, ['OCR → BiLSTM + GloVe 840B (300D)', 'Bidirectional context encoding'], const Color(0xFF34d399)),
        const SizedBox(height: 10),
        _stream('Fusion Layer', Icons.merge_type, ['Bahdanau Attention mechanism', 'Dynamic cross-modal alignment'], _kAccent),
      ]), bc: _kAccent.withOpacity(0.15)),
    ]);
  }

  Widget _stream(String t, IconData ic, List<String> pts, Color c) => Container(
    padding: const EdgeInsets.all(14),
    decoration: BoxDecoration(color: c.withOpacity(0.04), borderRadius: BorderRadius.circular(14), border: Border.all(color: c.withOpacity(0.12))),
    child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Row(children: [Icon(ic, color: c, size: 18), const SizedBox(width: 8), Text(t, style: GoogleFonts.outfit(color: c, fontSize: 14, fontWeight: FontWeight.w700))]),
      const SizedBox(height: 10),
      ...pts.map((p) => Padding(padding: const EdgeInsets.only(bottom: 5), child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Text('• ', style: GoogleFonts.inter(color: c.withOpacity(0.6), fontSize: 12)),
        Expanded(child: Text(p, style: GoogleFonts.inter(color: Colors.white.withOpacity(0.5), fontSize: 12, height: 1.4))),
      ]))),
    ]),
  );

  Widget _buildPipeline() {
    final steps = [('Input\nMeme', Icons.image, const Color(0xFF818cf8)), ('CNN +\nCLIP', Icons.remove_red_eye, const Color(0xFF60a5fa)), ('OCR →\nBiLSTM', Icons.text_snippet, const Color(0xFF34d399)), ('Bahdanau\nAttention', Icons.merge_type, _kAccent), ('Safe /\nToxic', Icons.shield, const Color(0xFFf87171))];
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      _sec('PROCESSING PIPELINE', Icons.account_tree_outlined),
      const SizedBox(height: 14),
      SizedBox(height: 120, child: ListView.separated(
        scrollDirection: Axis.horizontal, itemCount: steps.length,
        separatorBuilder: (_, __) => Padding(padding: const EdgeInsets.symmetric(vertical: 40), child: Icon(Icons.arrow_forward_ios, color: _kMuted.withOpacity(0.4), size: 14)),
        itemBuilder: (_, i) { final (l, ic, c) = steps[i]; return Container(width: 80, padding: const EdgeInsets.all(12), decoration: BoxDecoration(color: c.withOpacity(0.06), borderRadius: BorderRadius.circular(14), border: Border.all(color: c.withOpacity(0.15))),
          child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [Icon(ic, color: c, size: 24), const SizedBox(height: 8), Text(l, textAlign: TextAlign.center, style: GoogleFonts.firaCode(color: c, fontSize: 9, fontWeight: FontWeight.w600, height: 1.3))])); },
      )),
    ]);
  }

  Widget _buildTraining() {
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      _sec('3-STAGE TRAINING', Icons.school_outlined),
      const SizedBox(height: 14),
      _stage('01', 'Feature Alignment', 'CLIP priors frozen; CNN and BiLSTM learn spatial-textual relationships.', const Color(0xFF60a5fa)),
      const SizedBox(height: 10),
      _stage('02', 'Differential Fine-Tuning', 'Full architecture unfrozen with layer-specific learning rates.', const Color(0xFF818cf8)),
      const SizedBox(height: 10),
      _stage('03', 'Convergence Polish', 'Label smoothing + Cosine Annealing + Weighted Random Sampling.', _kAccent),
    ]);
  }

  Widget _stage(String n, String t, String d, Color c) => Container(
    padding: const EdgeInsets.all(16),
    decoration: BoxDecoration(color: _kSurface, borderRadius: BorderRadius.circular(16), border: Border.all(color: c.withOpacity(0.15))),
    child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Container(width: 36, height: 36, decoration: BoxDecoration(color: c.withOpacity(0.1), borderRadius: BorderRadius.circular(10)), alignment: Alignment.center, child: Text(n, style: GoogleFonts.outfit(color: c, fontSize: 14, fontWeight: FontWeight.w800))),
      const SizedBox(width: 14),
      Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Text(t, style: GoogleFonts.outfit(color: Colors.white, fontSize: 14, fontWeight: FontWeight.w700)),
        const SizedBox(height: 6),
        Text(d, style: GoogleFonts.inter(color: Colors.white.withOpacity(0.45), fontSize: 12, height: 1.55)),
      ])),
    ]),
  );

  Widget _buildWhyMatters() {
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      _sec('WHY IT MATTERS', Icons.lightbulb_outline),
      const SizedBox(height: 14),
      _glass(Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Text('Bridging Research & Deployment', style: GoogleFonts.outfit(color: Colors.white, fontSize: 16, fontWeight: FontWeight.w700)),
        const SizedBox(height: 12),
        Text('Current solutions rely on billion-parameter Transformers — computationally prohibitive for real-time moderation on mobile devices. Sentinel-X proves that an intelligently designed hybrid architecture can run complex multimodal operations locally.', style: GoogleFonts.inter(color: Colors.white.withOpacity(0.5), fontSize: 13, height: 1.6)),
      ])),
    ]);
  }

  Widget _buildTech() {
    final ts = [('PyTorch', Icons.memory), ('CLIP ViT-B/32', Icons.remove_red_eye), ('BiLSTM', Icons.swap_horiz), ('GloVe 300D', Icons.text_fields), ('CNN', Icons.grid_view), ('AdamW', Icons.tune), ('FastAPI', Icons.api), ('Flutter', Icons.phone_android), ('OCR', Icons.document_scanner)];
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      _sec('TECH STACK', Icons.code),
      const SizedBox(height: 14),
      Wrap(spacing: 8, runSpacing: 8, children: ts.map((t) => Container(
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
        decoration: BoxDecoration(color: _kSurface, borderRadius: BorderRadius.circular(20), border: Border.all(color: _kBorder)),
        child: Row(mainAxisSize: MainAxisSize.min, children: [Icon(t.$2, color: _kAccent, size: 14), const SizedBox(width: 6), Text(t.$1, style: GoogleFonts.firaCode(color: Colors.white.withOpacity(0.7), fontSize: 11, fontWeight: FontWeight.w500))]),
      )).toList()),
    ]);
  }

  Widget _sec(String t, IconData ic) => Row(children: [Icon(ic, color: _kAccent, size: 16), const SizedBox(width: 8), Text(t, style: GoogleFonts.outfit(color: _kAccent, fontSize: 12, fontWeight: FontWeight.w700, letterSpacing: 1.5))]);

  Widget _glass(Widget child, {Color? bc}) => ClipRRect(borderRadius: BorderRadius.circular(18), child: BackdropFilter(filter: ImageFilter.blur(sigmaX: 8, sigmaY: 8), child: Container(padding: const EdgeInsets.all(20), decoration: BoxDecoration(color: _kSurface.withOpacity(0.7), borderRadius: BorderRadius.circular(18), border: Border.all(color: bc ?? _kBorder.withOpacity(0.6))), child: child)));
}
