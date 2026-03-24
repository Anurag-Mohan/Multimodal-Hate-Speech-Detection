import 'package:flutter/material.dart';
import 'dart:ui';
import 'package:google_fonts/google_fonts.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'dart:math' as math;
import '../services/api_service.dart';

const kBg = Color(0xFF09090B);
const kSurface = Color(0xFF141416);
const kCard = Color(0xFF1C1C1F);
const kBorder = Color(0xFF27272A);
const kAccent = Color(0xFFC9A84C);
const kAccent2 = Color(0xFFF0D78C);
const kMuted = Color(0xFF71717A);
const kTermBg = Color(0xFF0C0C0E);

class _LogEntry {
  final String text;
  final _LogLevel level;
  _LogEntry(this.text, this.level);
}

enum _LogLevel { info, ok, warn, highlight }

const _rawLogs = [
  ('system', 'Sentinel-X v2.1 initialised', _LogLevel.highlight),
  ('io', 'Reading image buffer from device storage…', _LogLevel.info),
  ('io', 'Decoding pixel data (JPEG / PNG / WEBP)…', _LogLevel.info),
  ('io', 'Image decoded OK', _LogLevel.ok),
  ('preproc', 'Resizing to 224 × 224 for model input…', _LogLevel.info),
  ('preproc', 'Normalising pixel values  [μ=0.48, σ=0.26]', _LogLevel.info),
  ('preproc', 'Applying model preprocessing transform…', _LogLevel.info),
  ('preproc', 'Tensor shape confirmed: [1, 3, 224, 224]', _LogLevel.ok),
  ('ocr', 'Connecting to OCR.Space engine…', _LogLevel.info),
  ('ocr', 'Uploading image payload via HTTPS…', _LogLevel.info),
  ('ocr', 'OCR inference complete', _LogLevel.ok),
  ('ocr', 'Parsing & cleaning extracted text…', _LogLevel.info),
  ('model', 'Building multimodal analysis prompts…', _LogLevel.info),
  ('model', 'Tokenising prompts (max 77 tokens)…', _LogLevel.info),
  ('model', 'Encoding image tensor on device…', _LogLevel.info),
  ('model', 'Computing visual feature embeddings…', _LogLevel.info),
  ('model', 'Encoding text prompts with detection model…', _LogLevel.info),
  ('model', 'Feature embeddings  ‣  L2-normalised', _LogLevel.ok),
  ('score', 'Computing cosine similarity (img ↔ txt)…', _LogLevel.info),
  ('score', 'Applying softmax over [safe, toxic] logits…', _LogLevel.info),
  ('score', 'Running keyword-based text toxicity check…', _LogLevel.info),
  ('score', 'Blending detection score + text score via max()…', _LogLevel.info),
  ('score', 'Applying classification threshold  θ=0.50', _LogLevel.info),
  ('result', 'Generating final verdict…', _LogLevel.highlight),
];

class UploadScreen extends StatefulWidget {
  const UploadScreen({super.key});
  @override
  State<UploadScreen> createState() => _UploadScreenState();
}

class _UploadScreenState extends State<UploadScreen>
    with TickerProviderStateMixin {
  File? _image;
  bool _isScanning = false;
  Map<String, dynamic>? _scanResult;

  final List<_LogEntry> _logs = [];
  double _scanProgress = 0.0;

  late AnimationController _ringCtrl;
  late AnimationController _glowCtrl;
  late AnimationController _resultCtrl;
  late AnimationController _scanlineCtrl;
  late Animation<double> _resultFade;
  late Animation<Offset> _resultSlide;
  late Animation<double> _glowAnim;

  @override
  void initState() {
    super.initState();
    _ringCtrl = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 2),
    )..repeat();
    _glowCtrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1400),
    )..repeat(reverse: true);
    _glowAnim = Tween<double>(
      begin: 0.0,
      end: 1.0,
    ).animate(CurvedAnimation(parent: _glowCtrl, curve: Curves.easeInOut));
    _resultCtrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 700),
    );
    _resultFade = CurvedAnimation(parent: _resultCtrl, curve: Curves.easeOut);
    _resultSlide = Tween<Offset>(
      begin: const Offset(0, 0.12),
      end: Offset.zero,
    ).animate(CurvedAnimation(parent: _resultCtrl, curve: Curves.easeOutCubic));
    _scanlineCtrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 2200),
    );
  }

  @override
  void dispose() {
    _ringCtrl.dispose();
    _glowCtrl.dispose();
    _resultCtrl.dispose();
    _scanlineCtrl.dispose();
    super.dispose();
  }

  Future<void> _pickImage(ImageSource src) async {
    final f = await ImagePicker().pickImage(source: src);
    if (f != null) {
      setState(() {
        _image = File(f.path);
        _scanResult = null;
        _logs.clear();
        _scanProgress = 0;
      });
    }
  }

  void _showPickerSheet() {
    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      builder:
          (_) => Container(
            decoration: const BoxDecoration(
              color: kCard,
              borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
            ),
            child: SafeArea(
              child: Padding(
                padding: const EdgeInsets.symmetric(vertical: 16),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Container(
                      width: 36,
                      height: 4,
                      margin: const EdgeInsets.only(bottom: 16),
                      decoration: BoxDecoration(
                        color: kBorder,
                        borderRadius: BorderRadius.circular(2),
                      ),
                    ),
                    ListTile(
                      leading: Container(
                        width: 42,
                        height: 42,
                        decoration: BoxDecoration(
                          color: kAccent.withOpacity(0.08),
                          borderRadius: BorderRadius.circular(12),
                          border: Border.all(
                            color: kAccent.withOpacity(0.15),
                          ),
                        ),
                        child: const Icon(
                          Icons.photo_library_outlined,
                          color: kAccent,
                          size: 20,
                        ),
                      ),
                      title: Text(
                        'Photo Library',
                        style: GoogleFonts.inter(
                          color: Colors.white,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                      subtitle: Text(
                        'Choose existing photo',
                        style: GoogleFonts.inter(color: kMuted, fontSize: 12),
                      ),
                      onTap: () {
                        Navigator.pop(context);
                        _pickImage(ImageSource.gallery);
                      },
                    ),
                    ListTile(
                      leading: Container(
                        width: 42,
                        height: 42,
                        decoration: BoxDecoration(
                          color: kAccent.withOpacity(0.08),
                          borderRadius: BorderRadius.circular(12),
                          border: Border.all(
                            color: kAccent.withOpacity(0.15),
                          ),
                        ),
                        child: const Icon(
                          Icons.camera_alt_outlined,
                          color: kAccent,
                          size: 20,
                        ),
                      ),
                      title: Text(
                        'Camera',
                        style: GoogleFonts.inter(
                          color: Colors.white,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                      subtitle: Text(
                        'Take a new photo',
                        style: GoogleFonts.inter(color: kMuted, fontSize: 12),
                      ),
                      onTap: () {
                        Navigator.pop(context);
                        _pickImage(ImageSource.camera);
                      },
                    ),
                  ],
                ),
              ),
            ),
          ),
    );
  }

  Future<void> _initiateDeepScan() async {
    if (_image == null) return;
    setState(() {
      _isScanning = true;
      _logs.clear();
      _scanProgress = 0;
      _scanResult = null;
    });
    _scanlineCtrl.repeat();

    final total = _rawLogs.length;
    for (int i = 0; i < total; i++) {
      final delayMs =
          i < 4
              ? 180
              : i < 12
              ? 260
              : i < 18
              ? 320
              : 400;
      await Future.delayed(Duration(milliseconds: delayMs));
      if (!mounted) return;
      final (_, text, level) = _rawLogs[i];
      setState(() {
        _logs.add(_LogEntry(text, level));
        _scanProgress = (i + 1) / total;
      });
    }

    try {
      final result = await ApiService.scanMedia(_image!);
      if (!mounted) return;
      _scanlineCtrl.stop();
      setState(() {
        _scanResult = result;
        _isScanning = false;
      });
      _resultCtrl.forward(from: 0);
    } catch (e) {
      if (!mounted) return;
      _scanlineCtrl.stop();
      setState(() {
        _logs.add(_LogEntry('ERROR: $e', _LogLevel.warn));
        _isScanning = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: kBg,
      appBar: AppBar(
        backgroundColor: kBg.withOpacity(0.85),
        elevation: 0,
        surfaceTintColor: Colors.transparent,
        flexibleSpace: ClipRRect(
          child: BackdropFilter(
            filter: ImageFilter.blur(sigmaX: 20, sigmaY: 20),
            child: Container(color: Colors.transparent),
          ),
        ),
        leading: GestureDetector(
          onTap: () => Navigator.pop(context),
          child: const Icon(Icons.close, color: Colors.white70, size: 24),
        ),
        title: Text(
          'New Post',
          style: GoogleFonts.outfit(
            fontWeight: FontWeight.w600,
            fontSize: 18,
            color: Colors.white,
          ),
        ),
        centerTitle: true,
        actions: [
          if (_image != null && !_isScanning && _scanResult == null)
            TextButton(
              onPressed: _initiateDeepScan,
              child: ShaderMask(
                shaderCallback:
                    (b) => const LinearGradient(
                      colors: [kAccent, kAccent2],
                    ).createShader(b),
                child: Text(
                  'Scan',
                  style: GoogleFonts.outfit(
                    fontWeight: FontWeight.w700,
                    fontSize: 16,
                    color: Colors.white,
                  ),
                ),
              ),
            ),
          const SizedBox(width: 4),
        ],
        bottom: PreferredSize(
          preferredSize: const Size.fromHeight(0.3),
          child: Divider(
            color: kBorder.withOpacity(0.5),
            height: 0.3,
          ),
        ),
      ),
      body: SingleChildScrollView(
        physics: const BouncingScrollPhysics(),
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            _buildImageArea(),
            const SizedBox(height: 20),
            if (_image != null && !_isScanning && _scanResult == null)
              _buildScanButton(),
            if (_isScanning || _logs.isNotEmpty && _scanResult == null) ...[
              const SizedBox(height: 20),
              _buildScanningPanel(),
            ],
            if (_scanResult != null)
              FadeTransition(
                opacity: _resultFade,
                child: SlideTransition(
                  position: _resultSlide,
                  child: _buildResultCard(),
                ),
              ),
            const SizedBox(height: 48),
          ],
        ),
      ),
    );
  }

  Widget _buildImageArea() {
    return AnimatedBuilder(
      animation: _glowAnim,
      builder: (_, child) {
        return AnimatedContainer(
          duration: const Duration(milliseconds: 400),
          height: _image != null ? 320 : 240,
          decoration: BoxDecoration(
            color: kSurface,
            borderRadius: BorderRadius.circular(20),
            border: Border.all(
              color:
                  _isScanning
                      ? kAccent.withOpacity(0.15 + _glowAnim.value * 0.15)
                      : _image != null
                      ? kBorder.withOpacity(0.6)
                      : kBorder.withOpacity(0.4),
              width: 1,
            ),
            boxShadow:
                _isScanning
                    ? [
                      BoxShadow(
                        color: kAccent.withOpacity(0.06),
                        blurRadius: 30,
                        spreadRadius: 0,
                      ),
                    ]
                    : [],
          ),
          clipBehavior: Clip.antiAlias,
          child: child,
        );
      },
      child: _image != null ? _buildImagePreview() : _buildPickerHint(),
    );
  }

  Widget _buildImagePreview() {
    return Stack(
      fit: StackFit.expand,
      children: [
        Image.file(_image!, fit: BoxFit.cover),
        if (_isScanning)
          AnimatedBuilder(
            animation: _scanlineCtrl,
            builder: (_, __) {
              return Stack(
                children: [
                  Container(
                    color: Colors.black.withOpacity(0.15),
                  ),
                  Positioned(
                    top: _scanlineCtrl.value * 320,
                    left: 0,
                    right: 0,
                    child: Container(
                      height: 2,
                      decoration: BoxDecoration(
                        gradient: LinearGradient(
                          colors: [
                            Colors.transparent,
                            kAccent.withOpacity(0.6),
                            kAccent2.withOpacity(0.8),
                            kAccent.withOpacity(0.6),
                            Colors.transparent,
                          ],
                          stops: const [0, 0.2, 0.5, 0.8, 1],
                        ),
                        boxShadow: [
                          BoxShadow(
                            color: kAccent.withOpacity(0.3),
                            blurRadius: 12,
                            spreadRadius: 2,
                          ),
                        ],
                      ),
                    ),
                  ),
                  Center(
                    child: Text(
                      'ANALYSING',
                      style: GoogleFonts.outfit(
                        color: Colors.white.withOpacity(0.5),
                        fontSize: 11,
                        fontWeight: FontWeight.w600,
                        letterSpacing: 4,
                      ),
                    ),
                  ),
                ],
              );
            },
          ),
        if (!_isScanning && _scanResult == null)
          Positioned(
            bottom: 12,
            right: 12,
            child: GestureDetector(
              onTap: _showPickerSheet,
              child: ClipRRect(
                borderRadius: BorderRadius.circular(20),
                child: BackdropFilter(
                  filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
                  child: Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 14,
                      vertical: 8,
                    ),
                    decoration: BoxDecoration(
                      color: Colors.black.withOpacity(0.5),
                      borderRadius: BorderRadius.circular(20),
                      border: Border.all(
                        color: Colors.white.withOpacity(0.1),
                      ),
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        const Icon(
                          Icons.edit_outlined,
                          color: Colors.white70,
                          size: 13,
                        ),
                        const SizedBox(width: 5),
                        Text(
                          'Change',
                          style: GoogleFonts.inter(
                            color: Colors.white70,
                            fontSize: 12,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ),
          ),
      ],
    );
  }

  Widget _buildPickerHint() {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Container(
          width: 72,
          height: 72,
          decoration: BoxDecoration(
            color: kBorder.withOpacity(0.4),
            borderRadius: BorderRadius.circular(36),
            border: Border.all(color: kBorder.withOpacity(0.6)),
          ),
          child: Icon(
            Icons.add_photo_alternate_outlined,
            color: kMuted.withOpacity(0.7),
            size: 34,
          ),
        ),
        const SizedBox(height: 16),
        Text(
          'Select media to scan',
          style: GoogleFonts.outfit(
            color: Colors.white,
            fontSize: 16,
            fontWeight: FontWeight.w600,
          ),
        ),
        const SizedBox(height: 6),
        Text(
          'Choose from gallery or take a photo',
          style: GoogleFonts.inter(color: kMuted, fontSize: 13),
        ),
        const SizedBox(height: 24),
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            _miniBtn(
              Icons.photo_library_outlined,
              'Library',
              () => _pickImage(ImageSource.gallery),
            ),
            const SizedBox(width: 12),
            _miniBtn(
              Icons.camera_alt_outlined,
              'Camera',
              () => _pickImage(ImageSource.camera),
            ),
          ],
        ),
      ],
    );
  }

  Widget _miniBtn(IconData icon, String label, VoidCallback onTap) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 11),
        decoration: BoxDecoration(
          border: Border.all(color: kBorder),
          borderRadius: BorderRadius.circular(12),
          color: kSurface.withOpacity(0.5),
        ),
        child: Row(
          children: [
            Icon(icon, color: kAccent, size: 18),
            const SizedBox(width: 6),
            Text(
              label,
              style: GoogleFonts.inter(color: Colors.white, fontSize: 13),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildScanButton() {
    return Container(
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(16),
        gradient: const LinearGradient(
          colors: [kAccent, kAccent2],
          begin: Alignment.centerLeft,
          end: Alignment.centerRight,
        ),
        boxShadow: [
          BoxShadow(
            color: kAccent.withOpacity(0.2),
            blurRadius: 24,
            offset: const Offset(0, 8),
          ),
        ],
      ),
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          borderRadius: BorderRadius.circular(16),
          onTap: _initiateDeepScan,
          child: Padding(
            padding: const EdgeInsets.symmetric(vertical: 18),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                const Icon(
                  Icons.shield_outlined,
                  color: Color(0xFF09090B),
                  size: 22,
                ),
                const SizedBox(width: 10),
                Text(
                  'Run Sentinel Scan',
                  style: GoogleFonts.outfit(
                    color: const Color(0xFF09090B),
                    fontSize: 16,
                    fontWeight: FontWeight.w700,
                    letterSpacing: 0.3,
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildScanningPanel() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Row(
          children: [
            SizedBox(
              width: 64,
              height: 64,
              child: AnimatedBuilder(
                animation: _ringCtrl,
                builder:
                    (_, __) => CustomPaint(
                      painter: _ArcPainter(_ringCtrl.value, _scanProgress),
                    ),
              ),
            ),
            const SizedBox(width: 16),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  ShaderMask(
                    shaderCallback:
                        (b) => const LinearGradient(
                          colors: [kAccent, kAccent2],
                        ).createShader(b),
                    child: Text(
                      'Sentinel-X Analysis',
                      style: GoogleFonts.outfit(
                        color: Colors.white,
                        fontWeight: FontWeight.w700,
                        fontSize: 15,
                      ),
                    ),
                  ),
                  const SizedBox(height: 4),
                  ClipRRect(
                    borderRadius: BorderRadius.circular(4),
                    child: LinearProgressIndicator(
                      value: _scanProgress,
                      backgroundColor: kBorder,
                      valueColor: const AlwaysStoppedAnimation<Color>(kAccent),
                      minHeight: 4,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    '${(_scanProgress * 100).toStringAsFixed(0)}% complete',
                    style: GoogleFonts.inter(color: kMuted, fontSize: 11),
                  ),
                ],
              ),
            ),
          ],
        ),
        const SizedBox(height: 14),
        Container(
          height: 280,
          decoration: BoxDecoration(
            color: kTermBg,
            borderRadius: BorderRadius.circular(14),
            border: Border.all(color: kBorder.withOpacity(0.6)),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: 14,
                  vertical: 9,
                ),
                decoration: BoxDecoration(
                  color: const Color(0xFF0E0E10),
                  borderRadius: const BorderRadius.only(
                    topLeft: Radius.circular(13),
                    topRight: Radius.circular(13),
                  ),
                  border: Border(
                    bottom: BorderSide(color: kBorder.withOpacity(0.4)),
                  ),
                ),
                child: Row(
                  children: [
                    _dot(const Color(0xFFFF5F57)),
                    const SizedBox(width: 6),
                    _dot(const Color(0xFFFFBD2E)),
                    const SizedBox(width: 6),
                    _dot(const Color(0xFF28CA41)),
                    const SizedBox(width: 14),
                    Text(
                      'sentinel-x  —  inference log',
                      style: GoogleFonts.firaCode(
                        color: kMuted,
                        fontSize: 11,
                        fontWeight: FontWeight.w400,
                      ),
                    ),
                    const Spacer(),
                    if (_isScanning) _BlinkingDot(),
                  ],
                ),
              ),
              Expanded(
                child: _LogListView(logs: _logs, isRunning: _isScanning),
              ),
            ],
          ),
        ),
        const SizedBox(height: 12),
        _buildModuleBadges(),
      ],
    );
  }

  Widget _dot(Color c) => Container(
    width: 10,
    height: 10,
    decoration: BoxDecoration(color: c, shape: BoxShape.circle),
  );

  Widget _buildModuleBadges() {
    final phases = [
      ('I/O', _scanProgress >= 0.17),
      ('PREPROC', _scanProgress >= 0.33),
      ('OCR', _scanProgress >= 0.54),
      ('CLIP', _scanProgress >= 0.75),
      ('SCORE', _scanProgress >= 0.92),
    ];
    return Row(
      children:
          phases.map((p) {
            final done = p.$2;
            return Expanded(
              child: AnimatedContainer(
                duration: const Duration(milliseconds: 400),
                margin: const EdgeInsets.symmetric(horizontal: 3),
                padding: const EdgeInsets.symmetric(vertical: 7),
                decoration: BoxDecoration(
                  color: done ? kAccent.withOpacity(0.08) : kSurface,
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(
                    color: done ? kAccent.withOpacity(0.35) : kBorder.withOpacity(0.5),
                    width: 0.5,
                  ),
                ),
                child: Column(
                  children: [
                    Icon(
                      done ? Icons.check_circle : Icons.radio_button_unchecked,
                      color: done ? kAccent : kMuted.withOpacity(0.5),
                      size: 14,
                    ),
                    const SizedBox(height: 4),
                    Text(
                      p.$1,
                      style: GoogleFonts.firaCode(
                        fontSize: 9,
                        color: done ? kAccent : kMuted.withOpacity(0.5),
                        fontWeight: done ? FontWeight.w600 : FontWeight.w400,
                      ),
                      textAlign: TextAlign.center,
                    ),
                  ],
                ),
              ),
            );
          }).toList(),
    );
  }

  Widget _buildResultCard() {
    final bool isHateful = _scanResult?['is_hateful'] ?? false;
    final double prob = ((_scanResult?['prob_hateful'] ?? 0.0) as num) * 100;
    final String extracted = _scanResult?['extracted_text'] ?? '';
    final bool hasText =
        extracted.isNotEmpty &&
        extracted != 'NO_TEXT_DETECTED' &&
        extracted != 'OCR_FAILED';

    final statusColor =
        isHateful ? const Color(0xFFef4444) : const Color(0xFF22c55e);
    final statusBg =
        isHateful ? const Color(0xFF1a0505) : const Color(0xFF051a0d);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        const SizedBox(height: 8),
        Container(
          padding: const EdgeInsets.all(24),
          decoration: BoxDecoration(
            color: statusBg,
            borderRadius: const BorderRadius.only(
              topLeft: Radius.circular(20),
              topRight: Radius.circular(20),
            ),
            border: Border.all(color: statusColor.withOpacity(0.2)),
          ),
          child: Column(
            children: [
              Container(
                width: 78,
                height: 78,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: statusColor.withOpacity(0.08),
                  border: Border.all(
                    color: statusColor.withOpacity(0.2),
                    width: 1.5,
                  ),
                ),
                child: Icon(
                  isHateful ? Icons.warning_amber_rounded : Icons.verified_user,
                  color: statusColor,
                  size: 38,
                ),
              ),
              const SizedBox(height: 16),
              Text(
                isHateful ? 'Content Violation' : 'Content Approved',
                style: GoogleFonts.outfit(
                  color: statusColor,
                  fontSize: 24,
                  fontWeight: FontWeight.w700,
                ),
              ),
              const SizedBox(height: 8),
              Text(
                isHateful
                    ? 'This content violates community guidelines\nand cannot be posted.'
                    : 'All safety checks passed.\nThis content is cleared for sharing.',
                textAlign: TextAlign.center,
                style: GoogleFonts.inter(
                  color: Colors.white.withOpacity(0.45),
                  fontSize: 13,
                  height: 1.55,
                ),
              ),
            ],
          ),
        ),
        Container(
          padding: const EdgeInsets.fromLTRB(20, 20, 20, 20),
          decoration: BoxDecoration(
            color: kCard,
            border: Border(
              left: BorderSide(color: statusColor.withOpacity(0.2)),
              right: BorderSide(color: statusColor.withOpacity(0.2)),
            ),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'Analysis Results',
                style: GoogleFonts.outfit(
                  color: Colors.white,
                  fontWeight: FontWeight.w600,
                  fontSize: 14,
                ),
              ),
              const SizedBox(height: 16),
              Row(
                children: [
                  Text(
                    'Toxicity Score',
                    style: GoogleFonts.inter(color: kMuted, fontSize: 13),
                  ),
                  const Spacer(),
                  Text(
                    '${prob.toStringAsFixed(1)}%',
                    style: GoogleFonts.outfit(
                      color: statusColor,
                      fontWeight: FontWeight.w700,
                      fontSize: 15,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 8),
              ClipRRect(
                borderRadius: BorderRadius.circular(4),
                child: LinearProgressIndicator(
                  value: prob / 100,
                  backgroundColor: kBorder,
                  valueColor: AlwaysStoppedAnimation<Color>(statusColor),
                  minHeight: 6,
                ),
              ),
              const SizedBox(height: 4),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text(
                    'Safe',
                    style: GoogleFonts.firaCode(
                      color: const Color(0xFF4ade80),
                      fontSize: 10,
                    ),
                  ),
                  Text(
                    'Toxic',
                    style: GoogleFonts.firaCode(
                      color: const Color(0xFFf87171),
                      fontSize: 10,
                    ),
                  ),
                ],
              ),
              if (hasText) ...[
                const SizedBox(height: 20),
                Row(
                  children: [
                    Icon(Icons.text_fields, color: kMuted, size: 14),
                    const SizedBox(width: 6),
                    Text(
                      'Extracted Text (OCR)',
                      style: GoogleFonts.inter(color: kMuted, fontSize: 13),
                    ),
                  ],
                ),
                const SizedBox(height: 8),
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(13),
                  decoration: BoxDecoration(
                    color: kBg,
                    borderRadius: BorderRadius.circular(10),
                    border: Border.all(color: kBorder.withOpacity(0.5)),
                  ),
                  child: Text(
                    '"$extracted"',
                    style: GoogleFonts.inter(
                      color:
                          isHateful
                              ? const Color(0xFFfca5a5)
                              : const Color(0xFF86efac),
                      fontSize: 13,
                      fontStyle: FontStyle.italic,
                      height: 1.5,
                    ),
                  ),
                ),
              ],
            ],
          ),
        ),
        Container(
          decoration: BoxDecoration(
            color: kCard,
            borderRadius: const BorderRadius.only(
              bottomLeft: Radius.circular(20),
              bottomRight: Radius.circular(20),
            ),
            border: Border(
              left: BorderSide(color: statusColor.withOpacity(0.2)),
              right: BorderSide(color: statusColor.withOpacity(0.2)),
              bottom: BorderSide(color: statusColor.withOpacity(0.2)),
            ),
          ),
          padding: const EdgeInsets.fromLTRB(20, 4, 20, 20),
          child:
              isHateful
                  ? OutlinedButton.icon(
                    style: OutlinedButton.styleFrom(
                      foregroundColor: Colors.white60,
                      side: BorderSide(color: kBorder.withOpacity(0.6)),
                      padding: const EdgeInsets.symmetric(vertical: 14),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                    ),
                    icon: const Icon(Icons.delete_outline, size: 18),
                    label: Text(
                      'Discard Post',
                      style: GoogleFonts.inter(fontWeight: FontWeight.w600),
                    ),
                    onPressed: () => Navigator.pop(context),
                  )
                  : Container(
                    decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(12),
                      gradient: const LinearGradient(
                        colors: [kAccent, kAccent2],
                      ),
                      boxShadow: [
                        BoxShadow(
                          color: kAccent.withOpacity(0.2),
                          blurRadius: 20,
                          offset: const Offset(0, 6),
                        ),
                      ],
                    ),
                    child: Material(
                      color: Colors.transparent,
                      child: InkWell(
                        borderRadius: BorderRadius.circular(12),
                        onTap: () => Navigator.pop(context),
                        child: Padding(
                          padding: const EdgeInsets.symmetric(vertical: 15),
                          child: Row(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              const Icon(
                                Icons.send_rounded,
                                color: Color(0xFF09090B),
                                size: 18,
                              ),
                              const SizedBox(width: 8),
                              Text(
                                'Share to Feed',
                                style: GoogleFonts.outfit(
                                  color: const Color(0xFF09090B),
                                  fontSize: 15,
                                  fontWeight: FontWeight.w700,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),
                    ),
                  ),
        ),
      ],
    );
  }
}

class _ArcPainter extends CustomPainter {
  final double spin;
  final double progress;
  _ArcPainter(this.spin, this.progress);

  @override
  void paint(Canvas canvas, Size size) {
    final c = Offset(size.width / 2, size.height / 2);
    final r = size.width / 2 - 5;

    canvas.drawCircle(
      c,
      r,
      Paint()
        ..color = const Color(0xFF27272A)
        ..strokeWidth = 4
        ..style = PaintingStyle.stroke,
    );

    if (progress > 0) {
      canvas.drawArc(
        Rect.fromCircle(center: c, radius: r),
        -math.pi / 2,
        2 * math.pi * progress,
        false,
        Paint()
          ..color = kAccent.withOpacity(0.12)
          ..strokeWidth = 4
          ..style = PaintingStyle.stroke,
      );
    }

    final sweep = math.pi * 1.3;
    final start = spin * 2 * math.pi - math.pi / 2;
    final grad = SweepGradient(
      startAngle: start,
      endAngle: start + sweep,
      colors: const [kAccent2, kAccent],
    ).createShader(Rect.fromCircle(center: c, radius: r));

    canvas.drawArc(
      Rect.fromCircle(center: c, radius: r),
      start,
      sweep,
      false,
      Paint()
        ..shader = grad
        ..strokeWidth = 4
        ..strokeCap = StrokeCap.round
        ..style = PaintingStyle.stroke,
    );

    final pct = '${(progress * 100).toStringAsFixed(0)}%';
    final tp = TextPainter(
      text: TextSpan(
        text: pct,
        style: GoogleFonts.firaCode(
          color: Colors.white,
          fontSize: r * 0.42,
          fontWeight: FontWeight.w600,
        ),
      ),
      textDirection: TextDirection.ltr,
    )..layout();
    tp.paint(canvas, c - Offset(tp.width / 2, tp.height / 2));
  }

  @override
  bool shouldRepaint(_ArcPainter o) => o.spin != spin || o.progress != progress;
}

class _LogListView extends StatefulWidget {
  final List<_LogEntry> logs;
  final bool isRunning;
  const _LogListView({required this.logs, required this.isRunning});

  @override
  State<_LogListView> createState() => _LogListViewState();
}

class _LogListViewState extends State<_LogListView> {
  final ScrollController _sc = ScrollController();

  @override
  void didUpdateWidget(_LogListView old) {
    super.didUpdateWidget(old);
    if (widget.logs.length != old.logs.length) {
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (_sc.hasClients) {
          _sc.animateTo(
            _sc.position.maxScrollExtent,
            duration: const Duration(milliseconds: 250),
            curve: Curves.easeOut,
          );
        }
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return ListView.builder(
      controller: _sc,
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      itemCount: widget.logs.length + (widget.isRunning ? 1 : 0),
      itemBuilder: (_, i) {
        if (i == widget.logs.length) {
          return Padding(
            padding: const EdgeInsets.symmetric(vertical: 1),
            child: Row(
              children: [
                Text(
                  '> ',
                  style: GoogleFonts.firaCode(color: kAccent, fontSize: 11),
                ),
                _BlinkingCursor(),
              ],
            ),
          );
        }
        final entry = widget.logs[i];
        return _LogLine(entry: entry, index: i);
      },
    );
  }

  @override
  void dispose() {
    _sc.dispose();
    super.dispose();
  }
}

class _LogLine extends StatefulWidget {
  final _LogEntry entry;
  final int index;
  const _LogLine({required this.entry, required this.index});
  @override
  State<_LogLine> createState() => _LogLineState();
}

class _LogLineState extends State<_LogLine>
    with SingleTickerProviderStateMixin {
  late AnimationController _ctrl;
  late Animation<double> _fade;

  @override
  void initState() {
    super.initState();
    _ctrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 300),
    );
    _fade = CurvedAnimation(parent: _ctrl, curve: Curves.easeOut);
    _ctrl.forward();
  }

  @override
  Widget build(BuildContext context) {
    final e = widget.entry;
    Color col;
    String prefix;
    switch (e.level) {
      case _LogLevel.ok:
        col = const Color(0xFF4ade80);
        prefix = '✓ ';
        break;
      case _LogLevel.warn:
        col = const Color(0xFFf87171);
        prefix = '✗ ';
        break;
      case _LogLevel.highlight:
        col = kAccent;
        prefix = '► ';
        break;
      default:
        col = const Color(0xFF9CA3AF);
        prefix = '  ';
    }

    return FadeTransition(
      opacity: _fade,
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 1.5),
        child: RichText(
          text: TextSpan(
            style: GoogleFonts.firaCode(fontSize: 11, height: 1.4),
            children: [
              TextSpan(
                text: '${widget.index.toString().padLeft(2, '0')}  ',
                style: const TextStyle(color: Color(0xFF374151)),
              ),
              TextSpan(text: prefix, style: TextStyle(color: col)),
              TextSpan(text: e.text, style: TextStyle(color: col)),
            ],
          ),
        ),
      ),
    );
  }

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }
}

class _BlinkingCursor extends StatefulWidget {
  @override
  State<_BlinkingCursor> createState() => _BlinkingCursorState();
}

class _BlinkingCursorState extends State<_BlinkingCursor>
    with SingleTickerProviderStateMixin {
  late AnimationController _c;
  @override
  void initState() {
    super.initState();
    _c = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 550),
    )..repeat(reverse: true);
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _c,
      builder:
          (_, __) => Container(
            width: 7,
            height: 13,
            color: kAccent.withOpacity(_c.value),
          ),
    );
  }

  @override
  void dispose() {
    _c.dispose();
    super.dispose();
  }
}

class _BlinkingDot extends StatefulWidget {
  @override
  State<_BlinkingDot> createState() => _BlinkingDotState();
}

class _BlinkingDotState extends State<_BlinkingDot>
    with SingleTickerProviderStateMixin {
  late AnimationController _c;
  @override
  void initState() {
    super.initState();
    _c = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 700),
    )..repeat(reverse: true);
  }

  @override
  Widget build(BuildContext context) => AnimatedBuilder(
    animation: _c,
    builder:
        (_, __) => Container(
          width: 6,
          height: 6,
          decoration: BoxDecoration(
            color: kAccent.withOpacity(_c.value),
            shape: BoxShape.circle,
          ),
        ),
  );
  @override
  void dispose() {
    _c.dispose();
    super.dispose();
  }
}
