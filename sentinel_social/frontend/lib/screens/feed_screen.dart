import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'dart:ui';
import 'dart:io';
import 'package:google_fonts/google_fonts.dart';
import 'upload_screen.dart';
import 'about_screen.dart';
import 'model_screen.dart';
import '../services/feed_state.dart';

const kBg = Color(0xFF09090B);
const kCard = Color(0xFF141416);
const kSurface = Color(0xFF1C1C1F);
const kBorder = Color(0xFF27272A);
const kAccent = Color(0xFFC9A84C);
const kAccent2 = Color(0xFFF0D78C);
const kMuted = Color(0xFF71717A);

const _demoPosts = [
  {
    'user': 'alex.rivera',
    'avatar': 'https://i.pravatar.cc/150?img=11',
    'verified': true,
    'image': 'https://picsum.photos/seed/meme1/800/600',
    'likes': '12.4K',
    'caption': 'When the WiFi drops mid-Zoom call 😭 #relatable #wfh',
    'comments': 142,
    'time': '2 hours ago',
    'safe': true,
    'score': 8.3,
  },
  {
    'user': 'meme.central',
    'avatar': 'https://i.pravatar.cc/150?img=33',
    'verified': false,
    'image': 'https://picsum.photos/seed/meme2/800/600',
    'likes': '8.1K',
    'caption': 'Monday morning productivity vs Friday evening 😂 #mood',
    'comments': 89,
    'time': '5 hours ago',
    'safe': true,
    'score': 5.1,
  },
  {
    'user': 'sentinelx.ai',
    'avatar': 'https://i.pravatar.cc/150?img=57',
    'verified': true,
    'image': 'https://picsum.photos/seed/meme3/800/600',
    'likes': '31.2K',
    'caption': 'AI keeping the feed clean, one scan at a time 🛡️✨ #SentinelX',
    'comments': 301,
    'time': '1 day ago',
    'safe': true,
    'score': 3.0,
  },
];

class FeedScreen extends StatefulWidget {
  const FeedScreen({super.key});
  @override
  State<FeedScreen> createState() => _FeedScreenState();
}

class _FeedScreenState extends State<FeedScreen> {
  final Set<int> _liked = {};
  int _activeNav = 0;

  String _getTitle() {
    switch (_activeNav) {
      case 1: return 'About';
      case 3: return 'Model';
      default: return 'Sentinel';
    }
  }

  @override
  Widget build(BuildContext context) {
    return AnnotatedRegion<SystemUiOverlayStyle>(
      value: SystemUiOverlayStyle.light,
      child: Scaffold(
        backgroundColor: kBg,
        body: _activeNav == 0 ? _buildFeedBody() : _buildPageBody(),
        extendBody: true,
        bottomNavigationBar: _buildGlassNav(context),
      ),
    );
  }

  Widget _buildPageBody() {
    return CustomScrollView(slivers: [
      _buildAppBar(context),
      SliverFillRemaining(
        hasScrollBody: true,
        child: _activeNav == 1 ? const AboutScreen() : const ModelScreen(),
      ),
    ]);
  }

  Widget _buildFeedBody() {
    final feedState = FeedStateProvider.of(context);
    final userPosts = feedState.userPosts;
    final totalScans = feedState.scanHistory.length;
    return CustomScrollView(
      slivers: [
        _buildAppBar(context),
        SliverToBoxAdapter(child: _buildShieldBanner(totalScans, userPosts.length)),
        SliverToBoxAdapter(child: Divider(color: kBorder.withOpacity(0.5), thickness: 0.3, height: 0.3)),
        if (userPosts.isNotEmpty) ...[
          SliverList(delegate: SliverChildBuilderDelegate(
            (ctx, i) => _UserPostCard(post: userPosts[i]),
            childCount: userPosts.length,
          )),
        ],
        SliverList(delegate: SliverChildBuilderDelegate(
          (ctx, i) => _PostCard(
            post: _demoPosts[i],
            isLiked: _liked.contains(i),
            onLike: () => setState(() { _liked.contains(i) ? _liked.remove(i) : _liked.add(i); }),
          ),
          childCount: _demoPosts.length,
        )),
        const SliverToBoxAdapter(child: SizedBox(height: 100)),
      ],
    );
  }

  SliverAppBar _buildAppBar(BuildContext context) {
    return SliverAppBar(
      pinned: true,
      backgroundColor: kBg.withOpacity(0.85),
      surfaceTintColor: Colors.transparent,
      elevation: 0,
      flexibleSpace: ClipRRect(child: BackdropFilter(filter: ImageFilter.blur(sigmaX: 20, sigmaY: 20), child: Container(color: Colors.transparent))),
      title: ShaderMask(
        shaderCallback: (bounds) => const LinearGradient(colors: [kAccent, kAccent2], begin: Alignment.topLeft, end: Alignment.bottomRight).createShader(bounds),
        child: Text(_getTitle(), style: GoogleFonts.outfit(fontSize: 26, fontWeight: FontWeight.w700, color: Colors.white)),
      ),
      actions: [
        if (_activeNav == 0) ...[
          IconButton(icon: const Icon(Icons.add_box_outlined, size: 25), onPressed: () => _goUpload(), color: Colors.white70, tooltip: 'New Post'),
          Stack(children: [
            IconButton(icon: const Icon(Icons.favorite_border, size: 25), onPressed: () {}, color: Colors.white70),
            Positioned(right: 9, top: 9, child: Container(width: 7, height: 7, decoration: BoxDecoration(color: const Color(0xFFDC2626), shape: BoxShape.circle, border: Border.all(color: kBg, width: 1.5)))),
          ]),
          IconButton(icon: const Icon(Icons.send_outlined, size: 25), onPressed: () {}, color: Colors.white70),
        ],
        const SizedBox(width: 4),
      ],
      bottom: PreferredSize(preferredSize: const Size.fromHeight(0.3), child: Divider(color: kBorder.withOpacity(0.5), height: 0.3, thickness: 0.3)),
    );
  }

  void _goUpload() {
    Navigator.push(context, MaterialPageRoute(builder: (_) => const UploadScreen())).then((_) => setState(() {}));
  }

  Widget _buildShieldBanner(int totalScans, int approvedPosts) {
    return Container(
      margin: const EdgeInsets.fromLTRB(14, 10, 14, 10),
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(16),
        gradient: LinearGradient(
          begin: Alignment.centerLeft,
          end: Alignment.centerRight,
          colors: [kAccent.withOpacity(0.06), kSurface.withOpacity(0.5), const Color(0xFF22c55e).withOpacity(0.04)],
        ),
        border: Border.all(color: kAccent.withOpacity(0.1)),
      ),
      child: Row(children: [
        Container(
          width: 40, height: 40,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            gradient: LinearGradient(colors: [kAccent.withOpacity(0.15), kAccent.withOpacity(0.05)]),
            border: Border.all(color: kAccent.withOpacity(0.25)),
          ),
          child: const Icon(Icons.shield_rounded, color: kAccent, size: 20),
        ),
        const SizedBox(width: 14),
        Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Row(children: [
            Container(
              width: 6, height: 6,
              decoration: const BoxDecoration(color: Color(0xFF4ade80), shape: BoxShape.circle),
            ),
            const SizedBox(width: 6),
            Text('Sentinel Active', style: GoogleFonts.outfit(color: const Color(0xFF4ade80), fontSize: 12, fontWeight: FontWeight.w700)),
          ]),
          const SizedBox(height: 3),
          Text('All content AI-verified before reaching your feed', style: GoogleFonts.inter(color: kMuted, fontSize: 11)),
        ])),
        const SizedBox(width: 10),
        Column(children: [
          Text('$totalScans', style: GoogleFonts.outfit(color: kAccent, fontSize: 18, fontWeight: FontWeight.w800)),
          Text('scans', style: GoogleFonts.inter(color: kMuted, fontSize: 9)),
        ]),
        const SizedBox(width: 14),
        Column(children: [
          Text('$approvedPosts', style: GoogleFonts.outfit(color: const Color(0xFF4ade80), fontSize: 18, fontWeight: FontWeight.w800)),
          Text('posted', style: GoogleFonts.inter(color: kMuted, fontSize: 9)),
        ]),
      ]),
    );
  }

  Widget _buildGlassNav(BuildContext context) {
    return Container(
      padding: const EdgeInsets.only(bottom: 12, left: 16, right: 16),
      child: SafeArea(
        child: ClipRRect(
          borderRadius: BorderRadius.circular(28),
          child: BackdropFilter(
            filter: ImageFilter.blur(sigmaX: 32, sigmaY: 32),
            child: Container(
              height: 68,
              decoration: BoxDecoration(
                color: const Color(0xFF1A1A1E).withOpacity(0.75),
                borderRadius: BorderRadius.circular(28),
                border: Border.all(color: Colors.white.withOpacity(0.08), width: 0.8),
                boxShadow: [
                  BoxShadow(color: Colors.black.withOpacity(0.3), blurRadius: 20, offset: const Offset(0, 8)),
                  BoxShadow(color: kAccent.withOpacity(0.04), blurRadius: 30, spreadRadius: -5),
                ],
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  _glassNavItem(Icons.home_rounded, 'Home', 0),
                  _glassNavItem(Icons.info_outline_rounded, 'About', 1),
                  _uploadNavButton(),
                  _glassNavItem(Icons.insights_rounded, 'Model', 3),
                  _profileNavItem(),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _glassNavItem(IconData icon, String label, int index) {
    final active = _activeNav == index;
    return GestureDetector(
      onTap: () => setState(() => _activeNav = index),
      behavior: HitTestBehavior.opaque,
      child: SizedBox(
        width: 56,
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            AnimatedContainer(
              duration: const Duration(milliseconds: 200),
              padding: const EdgeInsets.all(6),
              decoration: BoxDecoration(
                color: active ? kAccent.withOpacity(0.12) : Colors.transparent,
                borderRadius: BorderRadius.circular(12),
              ),
              child: Icon(icon, color: active ? kAccent : Colors.white.withOpacity(0.4), size: 22),
            ),
            const SizedBox(height: 2),
            Text(label, style: GoogleFonts.inter(fontSize: 9, color: active ? kAccent : Colors.white.withOpacity(0.35), fontWeight: active ? FontWeight.w600 : FontWeight.w400)),
          ],
        ),
      ),
    );
  }

  Widget _uploadNavButton() {
    return GestureDetector(
      onTap: _goUpload,
      child: Container(
        width: 52, height: 52,
        margin: const EdgeInsets.only(bottom: 4),
        decoration: BoxDecoration(
          gradient: const LinearGradient(colors: [kAccent, kAccent2], begin: Alignment.topLeft, end: Alignment.bottomRight),
          shape: BoxShape.circle,
          boxShadow: [
            BoxShadow(color: kAccent.withOpacity(0.35), blurRadius: 16, offset: const Offset(0, 4)),
            BoxShadow(color: kAccent.withOpacity(0.15), blurRadius: 30, spreadRadius: 2),
          ],
        ),
        child: const Icon(Icons.add_rounded, color: Color(0xFF09090B), size: 28),
      ),
    );
  }

  Widget _profileNavItem() {
    final active = _activeNav == 4;
    return GestureDetector(
      onTap: () => setState(() => _activeNav = 4),
      behavior: HitTestBehavior.opaque,
      child: SizedBox(
        width: 56,
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Container(
              padding: const EdgeInsets.all(2),
              decoration: BoxDecoration(shape: BoxShape.circle, border: Border.all(color: active ? kAccent : Colors.transparent, width: 1.5)),
              child: CircleAvatar(radius: 13, backgroundImage: const NetworkImage('https://i.pravatar.cc/150?img=5'), backgroundColor: kSurface),
            ),
            const SizedBox(height: 2),
            Text('Profile', style: GoogleFonts.inter(fontSize: 9, color: active ? kAccent : Colors.white.withOpacity(0.35), fontWeight: active ? FontWeight.w600 : FontWeight.w400)),
          ],
        ),
      ),
    );
  }
}

class _UserPostCard extends StatelessWidget {
  final UserPost post;
  const _UserPostCard({required this.post});

  @override
  Widget build(BuildContext context) {
    final file = File(post.imagePath);
    final timeAgo = _timeAgo(post.timestamp);
    final scorePercent = (post.toxicityScore * 100).toStringAsFixed(1);
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Padding(
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 11),
        child: Row(children: [
          Container(
            padding: const EdgeInsets.all(2),
            decoration: const BoxDecoration(shape: BoxShape.circle, gradient: LinearGradient(colors: [kAccent, Color(0xFFE8C86A), kAccent2], begin: Alignment.bottomLeft, end: Alignment.topRight)),
            child: Container(
              padding: const EdgeInsets.all(1.5),
              decoration: const BoxDecoration(shape: BoxShape.circle, color: kBg),
              child: const CircleAvatar(radius: 17, backgroundImage: NetworkImage('https://i.pravatar.cc/150?img=5'), backgroundColor: kSurface),
            ),
          ),
          const SizedBox(width: 10),
          Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Row(children: [
              Text('you', style: GoogleFonts.inter(fontWeight: FontWeight.w600, fontSize: 13.5, color: Colors.white)),
              Padding(
                padding: const EdgeInsets.only(left: 4),
                child: ShaderMask(shaderCallback: (b) => const LinearGradient(colors: [kAccent, kAccent2]).createShader(b), child: const Icon(Icons.verified, color: Colors.white, size: 14)),
              ),
              const SizedBox(width: 6),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                decoration: BoxDecoration(color: kAccent.withOpacity(0.08), borderRadius: BorderRadius.circular(6)),
                child: Text('Scanned ✓', style: GoogleFonts.firaCode(fontSize: 8, color: kAccent, fontWeight: FontWeight.w600)),
              ),
            ]),
            Text(timeAgo, style: GoogleFonts.inter(fontSize: 11, color: kMuted)),
          ])),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 4),
            decoration: BoxDecoration(color: const Color(0xFF22c55e).withOpacity(0.08), borderRadius: BorderRadius.circular(20)),
            child: Row(mainAxisSize: MainAxisSize.min, children: [
              const Icon(Icons.shield, size: 10, color: Color(0xFF4ade80)),
              const SizedBox(width: 4),
              Text('Safe', style: GoogleFonts.inter(fontSize: 10, fontWeight: FontWeight.w600, color: const Color(0xFF4ade80))),
            ]),
          ),
          const SizedBox(width: 8),
          const Icon(Icons.more_horiz, color: Colors.white38, size: 20),
        ]),
      ),
      if (file.existsSync())
        Image.file(file, width: double.infinity, height: 380, fit: BoxFit.cover)
      else
        Container(width: double.infinity, height: 380, color: kSurface, child: const Center(child: Icon(Icons.broken_image, color: kMuted, size: 40))),
      Padding(
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 11),
        child: Row(children: [
          const Icon(Icons.favorite_border, color: Colors.white, size: 26),
          const SizedBox(width: 16),
          const Icon(Icons.chat_bubble_outline, color: Colors.white, size: 23),
          const SizedBox(width: 16),
          const Icon(Icons.send_outlined, color: Colors.white, size: 23),
          const Spacer(),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
            decoration: BoxDecoration(color: kSurface, borderRadius: BorderRadius.circular(8), border: Border.all(color: kBorder.withOpacity(0.5))),
            child: Text('Score: $scorePercent%', style: GoogleFonts.firaCode(fontSize: 10, color: const Color(0xFF4ade80))),
          ),
          const SizedBox(width: 8),
          const Icon(Icons.bookmark_border, color: Colors.white, size: 23),
        ]),
      ),
      if (post.caption.isNotEmpty) Padding(
        padding: const EdgeInsets.fromLTRB(14, 0, 14, 4),
        child: RichText(text: TextSpan(children: [
          TextSpan(text: 'you  ', style: GoogleFonts.inter(fontWeight: FontWeight.w600, fontSize: 13, color: Colors.white)),
          TextSpan(text: post.caption, style: GoogleFonts.inter(fontSize: 13, color: Colors.white.withOpacity(0.85))),
        ])),
      ),
      Padding(
        padding: const EdgeInsets.fromLTRB(14, 4, 14, 14),
        child: Text('Verified by Sentinel-X AI', style: GoogleFonts.inter(fontSize: 12.5, color: kMuted)),
      ),
      Divider(color: kBorder.withOpacity(0.4), height: 0.3, thickness: 0.3),
    ]);
  }

  static String _timeAgo(DateTime dt) {
    final diff = DateTime.now().difference(dt);
    if (diff.inMinutes < 1) return 'Just now';
    if (diff.inMinutes < 60) return '${diff.inMinutes}m ago';
    if (diff.inHours < 24) return '${diff.inHours}h ago';
    return '${diff.inDays}d ago';
  }
}

class _PostCard extends StatelessWidget {
  final Map<String, dynamic> post;
  final bool isLiked;
  final VoidCallback onLike;

  const _PostCard({required this.post, required this.isLiked, required this.onLike});

  @override
  Widget build(BuildContext context) {
    final bool safe = post['safe'] as bool? ?? true;
    final double score = (post['score'] as num?)?.toDouble() ?? 5.0;
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Padding(
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 11),
        child: Row(children: [
          Container(
            padding: const EdgeInsets.all(2),
            decoration: const BoxDecoration(shape: BoxShape.circle, gradient: LinearGradient(colors: [Color(0xFFC9A84C), Color(0xFFE8C86A), Color(0xFFF0D78C)], begin: Alignment.bottomLeft, end: Alignment.topRight)),
            child: Container(
              padding: const EdgeInsets.all(1.5),
              decoration: const BoxDecoration(shape: BoxShape.circle, color: kBg),
              child: CircleAvatar(radius: 17, backgroundImage: NetworkImage(post['avatar'] as String), backgroundColor: kSurface),
            ),
          ),
          const SizedBox(width: 10),
          Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Row(children: [
              Text(post['user'] as String, style: GoogleFonts.inter(fontWeight: FontWeight.w600, fontSize: 13.5, color: Colors.white)),
              if (post['verified'] as bool) Padding(
                padding: const EdgeInsets.only(left: 4),
                child: ShaderMask(shaderCallback: (b) => const LinearGradient(colors: [kAccent, kAccent2]).createShader(b), child: const Icon(Icons.verified, color: Colors.white, size: 14)),
              ),
            ]),
            Text(post['time'] as String, style: GoogleFonts.inter(fontSize: 11, color: kMuted)),
          ])),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 4),
            decoration: BoxDecoration(color: safe ? const Color(0xFF22c55e).withOpacity(0.08) : const Color(0xFFef4444).withOpacity(0.08), borderRadius: BorderRadius.circular(20)),
            child: Row(mainAxisSize: MainAxisSize.min, children: [
              Icon(safe ? Icons.shield : Icons.shield_outlined, size: 10, color: safe ? const Color(0xFF4ade80) : const Color(0xFFf87171)),
              const SizedBox(width: 4),
              Text(safe ? 'Safe' : 'Flagged', style: GoogleFonts.inter(fontSize: 10, fontWeight: FontWeight.w600, color: safe ? const Color(0xFF4ade80) : const Color(0xFFf87171))),
            ]),
          ),
          const SizedBox(width: 8),
          const Icon(Icons.more_horiz, color: Colors.white38, size: 20),
        ]),
      ),
      Stack(children: [
        Image.network(
          post['image'] as String, width: double.infinity, height: 380, fit: BoxFit.cover,
          loadingBuilder: (ctx, child, prog) => prog == null ? child : Container(
            width: double.infinity, height: 380, color: kSurface,
            child: Center(child: SizedBox(width: 24, height: 24, child: CircularProgressIndicator(color: kAccent.withOpacity(0.6), strokeWidth: 1.5))),
          ),
        ),
        Positioned(
          top: 10, left: 10,
          child: ClipRRect(
            borderRadius: BorderRadius.circular(8),
            child: BackdropFilter(
              filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                decoration: BoxDecoration(
                  color: Colors.black.withOpacity(0.4),
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: Colors.white.withOpacity(0.08)),
                ),
                child: Row(mainAxisSize: MainAxisSize.min, children: [
                  Icon(Icons.shield_rounded, size: 10, color: const Color(0xFF4ade80).withOpacity(0.8)),
                  const SizedBox(width: 4),
                  Text('${score.toStringAsFixed(1)}%', style: GoogleFonts.firaCode(fontSize: 9, color: Colors.white.withOpacity(0.7), fontWeight: FontWeight.w500)),
                ]),
              ),
            ),
          ),
        ),
      ]),
      Padding(
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 11),
        child: Row(children: [
          GestureDetector(
            onTap: onLike,
            child: AnimatedSwitcher(
              duration: const Duration(milliseconds: 200),
              transitionBuilder: (child, anim) => ScaleTransition(scale: anim, child: child),
              child: Icon(isLiked ? Icons.favorite : Icons.favorite_border, color: isLiked ? const Color(0xFFDC2626) : Colors.white, size: 26, key: ValueKey(isLiked)),
            ),
          ),
          const SizedBox(width: 16),
          const Icon(Icons.chat_bubble_outline, color: Colors.white, size: 23),
          const SizedBox(width: 16),
          const Icon(Icons.send_outlined, color: Colors.white, size: 23),
          const Spacer(),
          const Icon(Icons.bookmark_border, color: Colors.white, size: 23),
        ]),
      ),
      Padding(
        padding: const EdgeInsets.symmetric(horizontal: 14),
        child: Text('${isLiked ? "You and " : ""}${post["likes"]} likes', style: GoogleFonts.inter(fontWeight: FontWeight.w600, fontSize: 13, color: Colors.white)),
      ),
      Padding(
        padding: const EdgeInsets.fromLTRB(14, 4, 14, 2),
        child: RichText(text: TextSpan(children: [
          TextSpan(text: '${post["user"]}  ', style: GoogleFonts.inter(fontWeight: FontWeight.w600, fontSize: 13, color: Colors.white)),
          TextSpan(text: post['caption'] as String, style: GoogleFonts.inter(fontSize: 13, color: Colors.white.withOpacity(0.85))),
        ])),
      ),
      Padding(
        padding: const EdgeInsets.fromLTRB(14, 4, 14, 14),
        child: Row(children: [
          Text('View all ${post["comments"]} comments', style: GoogleFonts.inter(fontSize: 12.5, color: kMuted)),
          const Spacer(),
          Icon(Icons.verified_user_outlined, size: 11, color: kMuted.withOpacity(0.5)),
          const SizedBox(width: 3),
          Text('AI-verified', style: GoogleFonts.firaCode(fontSize: 9, color: kMuted.withOpacity(0.5))),
        ]),
      ),
      Divider(color: kBorder.withOpacity(0.4), height: 0.3, thickness: 0.3),
    ]);
  }
}
